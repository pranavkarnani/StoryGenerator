# -*- coding: utf-8
import numpy as np
import torch.nn as nn
import pandas as pd
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import StoryOutlineDataset
from torch.utils.data import random_split
from datetime import datetime
import os.path as osp
from path.path import get_project_root
import os, shutil, pickle, json, random, torch, wandb, copy, csv
from pprint import pprint
from datasets import load_metric
from typing import List
from utils.evaluation import COCOEvalCap
from get_embeddings_from_bert import sentence_encoder
from params import args         # Hyper-parameters
from setproctitle import *

setproctitle('윤석,윤호야잠시만쓸게')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#print(os.getcwd())
if args.debug:
    os.environ["WANDB_MODE"] = "offline"
wandb.init(project="project4", entity="cmu_idl_story_generation", name="server217_ADDBERT")
wandb.config = vars(args)

def train_val_split(split, dataset):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    return train_size, val_size

def get_tokenizer(model_name="gpt2", special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("  >> {} tokenizer is loaded...".format(model_name))
    if special_tokens:
        nums = tokenizer.add_special_tokens(special_tokens)
        print("  >> {} special tokens are added".format(nums))

    return tokenizer

def get_model(tokenizer, model_name="gpt2", special_tokens=None, load_model_path=None, parallel=False):
    if special_tokens:
        config = AutoConfig.from_pretrained(model_name,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)

    else:
        config = AutoConfig.from_pretrained(model_name,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)
    #model = AutoModel.from_config(config)
    model = AutoModelForCausalLM.from_config(config)
    print("  >> Pretrained Model {} loaded...".format(model_name))
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        print("  >> model loaded from {}".format(load_model_path))

    if parallel is True:
        if torch.cuda.device_count() > 1:
            print("  >> Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

    model.cuda()
    return model



def main():

    # Load Tokenizer
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>'}
    tokenizer = get_tokenizer(model_name=args.model_name, special_tokens=special_tokens_dict)

    print("  >> Total vocab size: {}".format(len(tokenizer)))
    print("  >> <BOS> Token ID: {}".format(tokenizer.encode("<BOS>")))
    print("  >> <SEP> Token ID: {}".format(tokenizer.encode("<SEP>")))
    print("  >> <EOS> Token ID: {}".format(tokenizer.encode("<EOS>")))
    print("")

    # Read Data
    data = pd.read_csv(args.data_path)
    data = data.dropna()
    data = data.reset_index()

    # Build Dataloader
    story_dataset = StoryOutlineDataset(data.loc[0:args.dataset_size], tokenizer, args.max_seq_length)
    train_size, val_size = train_val_split(0.9, story_dataset)
    train_dataset, val_dataset = random_split(story_dataset, [train_size, val_size])
    test_dataset = copy.deepcopy(val_dataset)        # test_dataset == val_dataset
    test_dataset.dataset.inference_flag = True
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    test_dataset.dataset.tokenizer.padding_side = "left"
    test_dataset.dataset.tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.dataset.collate_fn,
                              #num_workers=4,
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            pin_memory=True, shuffle=False,
                            collate_fn=val_dataset.dataset.collate_fn,
                            #num_workers = 4,
                            )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             pin_memory=True, shuffle=False,
                             collate_fn=test_dataset.dataset.collate_fn,
                             # num_workers = 4,
                             )

    # get pretrained model
    model = get_model(tokenizer=tokenizer, model_name=args.model_name,
                      special_tokens=special_tokens_dict, parallel=args.parallel)

    sentence_encoder_model = sentence_encoder(model_name=args.sentence_encoder_name)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    total_steps = len(train_loader) * args.n_epochs
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,
                                                                     T_mult=3,
                                                                 eta_min=1e-7)
    """
    """
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=int(len(train_loader) * 0.5),
                                                             num_training_steps=total_steps,)
    """
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=total_steps,)
    scaler = torch.cuda.amp.GradScaler()
    mse_loss = nn.MSELoss()

    # Training, Validation, Inference
    save_dir_path = prepare_training(args=wandb.config)

    for epoch_i in range(0, args.n_epochs):
        print(f'Epoch {epoch_i + 1} of {args.n_epochs}')
        model, avg_train_loss = train(ep=epoch_i, scaler=scaler, train_loader=train_loader,
                                      model=model, optimizer=optimizer, scheduler=scheduler,
                                      sentence_encoder_model=sentence_encoder_model)
        avg_val_loss = validate(model, val_loader, optimizer, tokenizer)
        eval_score, preds_list, labels_list, outline_list = inference(
            model=model, tokenizer=test_dataset.dataset.tokenizer, test_dataloader=test_loader)
        save(save_dir_path=save_dir_path, cur_epoch=epoch_i, model=model,
             valid_loss=avg_val_loss, eval_score=eval_score, outline_list=outline_list,
             preds_list=preds_list, labels_list=labels_list, best_result_flag=False)

def format_out_texts(text, tokenizer):
    t_map = tokenizer.special_tokens_map
    for key in t_map:
        text = text.replace(t_map[key], '')
    return text


def decoding_strategy(method:str, model, input_ids, attn_mask):
    if method.lower() == "nucleus":
        # Top-p (nucleus) text generation (10 samples):
        sample_outputs = model.generate(input_ids,
                                        attention_mask=attn_mask,
                                        do_sample=True,
                                        min_length=20,
                                        max_length=args.max_generation_length,
                                        top_k=0,
                                        top_p=0.92,
                                        temperature=0.9,
                                        repetition_penalty=2.0,
                                        num_return_sequences=1
                                        )

    return sample_outputs

def inference(model, tokenizer, test_dataloader):
    # Batch Inference
    model.eval()
    preds_list = []
    outline_list = []
    labels_list = []
    num_steps = len(test_dataloader)
    batch_bar = tqdm(total=num_steps, dynamic_ncols=True, leave=False, position=0, desc='Inference')
    for i, batch in enumerate(test_dataloader):
        if i == 50:
            break
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        labels = batch[2]

        sample_outputs = decoding_strategy(method=args.decoding_name, model=model,
                                           input_ids=input_ids, attn_mask=attn_mask)

        input_ids = input_ids.detach().cpu().numpy()
        sample_outputs = sample_outputs.detach().cpu().numpy()

        for i, (input_id, sample_output) in enumerate(zip(input_ids, sample_outputs)):

            gen_idx = np.where(sample_output == tokenizer.sep_token_id)[0][0]
            text = tokenizer.decode(sample_output[gen_idx:], skip_special_tokens=True)
            outline = tokenizer.decode(input_id, skip_special_tokens=True)
            #a = len(title) + len(','.join(keywords))
            #pred = text[len(outline):]
            outline_list.append(outline)
            preds_list.append(text)
            labels_list.append(labels[i])
            #print("{}: {}\n\n".format(i + 1, text[len(outline):]))

        batch_bar.update()
    batch_bar.close()

    automatic_evaluation = COCOEvalCap(preds=preds_list, refs=labels_list)
    automatic_evaluation.evaluate()
    print('-' * 25)
    print("| # of test stories: {}".format(len(preds_list)))
    for metric, score in automatic_evaluation.eval.items():
        print('| %s: %2.4f' % (metric, score))
        key_value = "inference " + str(metric)
        wandb.log({key_value: score})
        #wandb.log({"inference {}: {}".format(metric, score)})
    print('-' * 25)

    return automatic_evaluation.eval, preds_list, labels_list, outline_list

def _inference(model, tokenizer, val_loader):
    model.eval()

    for i, batch in enumerate(val_loader):

        if i % 100 == 0:
            lens = np.array([])
            input_ids = batch[0].numpy()
            attn_masks = batch[1].numpy()

            truncated_input = []
            truncated_attention_mask = []
            for i, input_id in enumerate(input_ids):
                context_index = np.where(input_id == 50260)[0][0]
                truncated_input.append(input_id[:context_index + 1])
                truncated_attention_mask.append(attn_masks[i][:context_index + 1])
                lens = np.append(lens, context_index + 1)

            max_len = int(np.amax(lens))

            padded_tokens = []
            for tok_ids in truncated_input:
                padded_tokens.append(list(tok_ids) + [0] * (max_len - len(tok_ids)))

            padded_tokens = torch.LongTensor(padded_tokens).cuda()
            attn_mask = np.zeros(padded_tokens.shape)

            for ix, lengths in enumerate(lens):
                #print(ix)
                #print(lengths)
                attn_mask[ix][:int(lengths)] = 1

            attn_mask = torch.tensor(attn_mask).long().cuda()

    story_ids = model.generate(padded_tokens, attention_mask=attn_mask,
                               num_beams=5,
                               max_length=800,
                               temperature=0.9,
                               remove_invalid_values=True,
                               top_k=50,
                               do_sample=True)

    raw_stories = [tokenizer.decode(story) for story in story_ids]
    #output_texts = list(map(format_out_texts, (raw_stories, tokenizer)))

    output_texts = [format_out_texts(story, tokenizer) for story in raw_stories]

    print(output_texts)
    return output_texts

def get_generated_sentences(outputs:np.ndarray, index_list: List, generator_tokenizer):
    """
    :param outputs: batch, max_seq
    :param index_list: (batch_size, )
    :return:
    """
    gen_tokens = [outputs[idx][point+1:] for idx, point in enumerate(index_list)]
    gen_texts = [generator_tokenizer.decode(toks, skip_special_tokens=True) for toks in gen_tokens]
    return gen_texts



def train(ep, scaler, train_loader, model, optimizer, scheduler, sentence_encoder_model):
    total_train_loss = 0
    num_steps = len(train_loader)
    tokenizer = train_loader.dataset.dataset.tokenizer
    batch_bar = tqdm(total=num_steps, dynamic_ncols=True, leave=False, position=0, desc='Train')
    for step, batch in enumerate(tqdm(train_loader)):

        model.train()

        b_input_ids = batch[0]
        b_masks = batch[1].cuda()

        labels = b_input_ids.clone().numpy()
        context_index_list = []
        for i, text in enumerate(b_input_ids.numpy()):
            context_index = np.where(text == 50260)[0][0]
            context_index_list.append(context_index)
            labels[i][:context_index + 1] = -100

        model.zero_grad()

        b_input_ids = b_input_ids.cuda()
        labels = torch.tensor(labels).cuda()

        with torch.cuda.amp.autocast():

            outputs = model(b_input_ids,
                            attention_mask=b_masks,
                            labels=labels,
                            token_type_ids=None)

            # Loss1: Cross-entropy loss
            loss1 = outputs.loss.mean()


            pred = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
            gen_stories =  get_generated_sentences(outputs=pred, index_list=context_index_list, generator_tokenizer=tokenizer)
            target_stories = get_generated_sentences(outputs=labels.detach().cpu().numpy(),
                                                     index_list=context_index_list, generator_tokenizer=tokenizer)
            # CLS Embeddings of "generated stories" and "target stories"
            gen_embeds = sentence_encoder_model.get_embeddings(gen_stories)
            tar_embeds = sentence_encoder_model.get_embeddings(target_stories)

            cos = nn.CosineSimilarity(dim=1)
            loss2 = cos(gen_embeds, tar_embeds).mean()
            #print(loss2)
            #print("PAUSE")
            loss = loss1 + loss2        # Joint Loss

        batch_loss = loss
        wandb.log({"training loss": batch_loss.item()})

        total_train_loss += batch_loss.item()
        batch_bar.set_postfix(
            loss="{:.06f}".format(float(total_train_loss / (step + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_bar.update()

    avg_train_loss = total_train_loss / len(train_loader)
    batch_bar.close()
    print("\n  >> Epoch {}: Train Loss {:.06f}, Learning Rate {:.06f}".format(
        ep,
        float(avg_train_loss),
        float(optimizer.param_groups[0]['lr'])))
    #print(f'Average Training Loss: {avg_train_loss}.')
    return model, avg_train_loss


def validate(model, val_dataloader, optimizer, tokenizer):
    model.eval()
    total_eval_loss = 0

    num_steps = len(val_dataloader)
    batch_bar = tqdm(total=num_steps, dynamic_ncols=True, leave=False, position=0, desc='Validation')
    for idx, batch in enumerate(val_dataloader):
        b_input_ids = batch[0]
        b_masks = batch[1].cuda()

        labels = b_input_ids.clone().numpy()

        for i, text in enumerate(b_input_ids.numpy()):
            context_index = np.where(text == 50260)[0][0]
            labels[i][:context_index + 1] = -100

        b_input_ids = b_input_ids.cuda()
        labels = torch.tensor(labels).cuda()

        with torch.no_grad():
            outputs = model(b_input_ids,
                            attention_mask=b_masks,
                            labels=labels)

            loss = outputs.loss.mean()

        batch_loss = loss
        #wandb.log({"validation loss": batch_loss.item()})
        total_eval_loss += batch_loss.item()
        batch_bar.set_postfix(
            loss="{:.06f}".format(float(total_eval_loss / (idx + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()

    avg_val_loss = total_eval_loss / len(val_dataloader)
    wandb.log({"validation loss": avg_val_loss})
    batch_bar.close()
    print("\n  >> validation: valid Loss {:.06f}".format(float(avg_val_loss)))

    #output_texts = inference(model, tokenizer, val_dataloader)

    #torch.save(model.state_dict(), '/content/' + file_name)
    return avg_val_loss

def prepare_training(args):
    # SAVE PATH
    date_string = datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = osp.join(osp.join(get_project_root(), "save"), date_string)
    src_dir_path = osp.join(save_dir_path, "src")
    model_dir_path = osp.join(save_dir_path, "model")
    if not osp.isdir(save_dir_path):
        os.makedirs(save_dir_path, exist_ok=True)

    if not osp.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)

    if not osp.isdir(src_dir_path):
        os.makedirs(src_dir_path, exist_ok=True)


    # Copy Source Code
    filelist = ["all.py", "dataset.py"]
    dirlist = ["path"]
    for filename in filelist:
        source = osp.join(get_project_root(), filename)
        shutil.copy(source, src_dir_path)
    for dirname in dirlist:
        source = osp.join(get_project_root(), dirname)
        dest = osp.join(src_dir_path, dirname)
        shutil.copytree(source, dest)
    print("  >> src_dir_path: {}".format(src_dir_path))

    print("  >> save_dir_path: {}".format(save_dir_path))
    # Save Hyper-parameters
    args_filename = osp.join(save_dir_path, "args.bin")
    print("  >> save_hparams_path: {}".format(args_filename))

    with open(args_filename, "wb") as f:
        pickle.dump(args, f)
    print("  >> Hyper-Parameters:")
    #pprint(vars(args))
    pprint(args)
    print("")

    return save_dir_path

def save(save_dir_path, cur_epoch, model, valid_loss, eval_score,
         outline_list, preds_list, labels_list, best_result_flag=False):
    model_dir = osp.join(save_dir_path, "model")
    dir_path = osp.join(model_dir, str(cur_epoch))
    if not osp.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    model_name = "model_"+str(cur_epoch)+".pt"
    filepath = osp.join(dir_path, model_name)
    #torch.save(model, filepath)
    torch.save(model.state_dict(), filepath)

    print("  >> {}-steps model saved...{}".format(cur_epoch, filepath))
    filepath = osp.join(dir_path, "valid_result.json")
    eval_score["valid_loss"] = valid_loss
    eval_score["current_epopch"] = cur_epoch
    with open(filepath, "w") as f:
        #json.dump({"valid_loss": valid_loss,
        #           "current_epoch": cur_epoch}, f)
        json.dump(eval_score, f)

    # Save Inference Result
    fieldnames = ["index", "outline", "generated_stories", "gold-standard_stories"]
    inference_output_filepath = osp.join(dir_path, "inference_examples "+str(len(preds_list))+".csv")
    with open(inference_output_filepath, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, (outline, pred, gold) in enumerate(zip(outline_list, preds_list, labels_list)):
            elem_dict = {"index": index,
                         "outline": outline,
                         "generated_stories": pred,
                         "gold-standard_stories": gold}
            writer.writerow(elem_dict)

    print("  {} of Generated Stories are saved into {}".format(len(preds_list), inference_output_filepath))

    if best_result_flag is True:
        best_path = osp.join(save_dir_path, "best.json")
        with open(best_path, "w") as f:
            #json.dump({"best_loss": valid_loss,
            #           "current_epoch": cur_epoch}, f)
            json.dump(eval_score, f)




    return


if __name__ == "__main__":
    main()