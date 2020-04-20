import re
import pandas as pd
import bs4
import requests
import spacy
import matplotlib
from spacy import displacy
import numpy as np
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)

candidate_sentences = pd.read_csv("D:\\tech_title.csv",encoding='unicode_escape')
candidate_sentences.shape
print(candidate_sentences['content'].sample(5))
def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################

  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text

      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text

      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text

      ## chunk 5
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

entity_pairs = []

for i in tqdm(candidate_sentences["content"]):
  entity_pairs.append(get_entities(i))

print(entity_pairs[10:20])
def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object
  matcher = Matcher(nlp.vocab)

  #define the pattern
  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

  matcher.add("matching_1", None, pattern)

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]]

  return(span.text)


relations = [get_relation(i) for i in tqdm(candidate_sentences['content'])]
print(pd.Series(relations).value_counts()[:50])

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
source = [i[0] for i in entity_pairs]
# extract object
target = [i[1] for i in entity_pairs]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df, "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(16,16))
pos = nx.spring_layout(G)
# nx.draw_networkx(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos,node_size=400,font_size=15,node_shape='o')
nx.draw_shell(G,with_labels=True)
print(nx.adjacency_matrix(G))
g=nx.adjacency_data(G)
# print(g)
n=nx.to_numpy_matrix(G)
l=nx.to_dict_of_lists(G)
print(l)
print(nx.to_dict_of_dicts(G))
print(nx.to_edgelist(G))
# print(nx.to_pandas_adjacency(G))
# np.savetxt("AM.txt",n,delimiter=",",newline="\n",fmt="%d")
plt.show()


