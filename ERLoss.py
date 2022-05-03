import spacy
from spacy import displacy

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import nltk
nltk.download("punkt")

from nltk.tokenize import word_tokenize

nlp = spacy.load('en_core_web_sm')

def get_entity_pairs(sentences):
    entity_pairs = []
    for i in sentences:
        entity_pairs.append(get_entities(i))
    return entity_pairs

def get_entities(sent):

  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    
  prv_tok_text = ""   

  prefix = ""
  modifier = ""
  
  for tok in nlp(sent):
    
    if tok.dep_ != "punct":
      
      if tok.dep_ == "compound":
        prefix = tok.text
        
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      

      if tok.dep_.endswith("mod") == True:
        modifier = tok.text

        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text

  return [ent1.strip(), ent2.strip()]


def get_relation(sent):

  doc = nlp(sent)

  matcher = Matcher(nlp.vocab)

  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", None, pattern) 

  matches = matcher(doc)
  k = len(matches) - 1
  if len(doc) > k and k > 0 and k is not None:
    span = doc[matches[k][1]:matches[k][2]] 
    return(span.text)
  else:
    return ''

def get_relations(sentences):
    relations = [get_relation(i) for i in sentences]
    return relations


def get_er(story):
    sentences = story.split(".")
    entity_pairs = get_entity_pairs(sentences)
    relations = get_relations(sentences)
    sequence = ""
    for i in range(len(entity_pairs)):
        if relations[i] != '':
            sequence += entity_pairs[i][0] + ' '
            sequence += relations[i] + ' '
            sequence += entity_pairs[i][1] + ' '
    return sequence

