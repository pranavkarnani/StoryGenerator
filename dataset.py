#-*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class StoryOutlineDataset(Dataset):

    def __init__(self, data, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.data = data
        self.labels_attn = []
        self.input_data_list = []
        self.text_label_list = []
        self.inference_flag = False

        for i in tqdm(range(len(self.data))):
            text = self.data.loc[i, 'text'].replace("\n", " ")
            outline = self.data.loc[i, 'storyline'].split(' ')
            outline = " ".join(outline[:100]).replace("<SEP>", "")

            input = tokenizer.bos_token + outline + tokenizer.sep_token
            text_label =  text + tokenizer.eos_token

            """
            encodings_dict_story = tokenizer('<BOS> ' + input + ' <EOS>',
                                             truncation=True,
                                             max_length=max_input_length,
                                             padding=True
                                             )
            """
            self.input_data_list.append(input)
            self.text_label_list.append(text_label)
        assert len(self.input_data_list) == len(self.text_label_list)
        print("  >> Complete loading the whole data")
        #self.input_ids.append(torch.tensor(encodings_dict_story['input_ids']))
        #self.attn_masks.append(torch.tensor(encodings_dict_story['attention_mask']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):

        input_str = self.input_data_list[ind]
        # Training
        if self.inference_flag is False:
            input_str += self.text_label_list[ind]

        return input_str, self.text_label_list[ind]


    def collate_fn(self, batch):

        # Due to inference
        batch_inputs = [x for x, y in batch]
        batch_labels = [y for x, y in batch]
        #inference_flag = batch[0][1]

        if self.inference_flag:

            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            encodings_dict_story = self.tokenizer(batch_inputs,
                                                  #truncation=True,
                                                  #max_length=self.max_seq_length,
                                                  padding=True)

            input_ids = torch.tensor(encodings_dict_story['input_ids'])
            attn_masks = torch.tensor(encodings_dict_story['attention_mask'])

            return input_ids, attn_masks, batch_labels


        else:
            encodings_dict_story = self.tokenizer(batch_inputs,
                                                  truncation=True,
                                                  max_length=self.max_seq_length,
                                                  padding=True)

        input_ids = torch.tensor(encodings_dict_story['input_ids'])
        attn_masks = torch.tensor(encodings_dict_story['attention_mask'])

        return input_ids, attn_masks