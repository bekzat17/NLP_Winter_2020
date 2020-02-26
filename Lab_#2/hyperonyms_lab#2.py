#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import wikipedia
import os  
import json 
from yargy import Parser, rule, and_, or_, not_ 
from yargy.interpretation import fact, attribute 
from yargy.predicates import gram, is_capitalized, dictionary, eq
import re 
from tqdm import tqdm_notebook 
from gensim import utils
from collections import defaultdict

wikipedia.set_lang("ru")
DATA_PATH_LIST = ['.']
EMBEDDING_MODEL_FILENAME = "wiki_node2vec.bin"
DATA_PATH="/".join(DATA_PATH_LIST+["training_nouns.tsv"])
df = pd.read_csv(DATA_PATH,sep='\t')


# In[2]:


def prestr(x):
    return str(x).replace('\"','').replace("'",'"')


# In[3]:


class DefDict(defaultdict):
    def __missing__(self, key):
        self[key] = key
        return key
    
idx2syns = DefDict(lambda x:x)
for val in df.values:
    idx2syns[val[0]]=val[1]
    try:
        pidxs = json.loads(prestr(val[2]))
        concp = [el.split(",")[0] for el in json.loads(prestr(val[3]))]
        idx2syns.update(dict(zip(pidxs,concp)))
    except:
        print(prestr(val[2]))
        print(prestr(val[3]))


# In[ ]:





# In[4]:


START = rule(
    or_(
        rule(gram('ADJF')),
        rule(gram('NOUN'))
    ).optional(),
    gram('NOUN')
)

START_S = or_(
    eq('такой'),
    eq('такие'),
)

KAK = eq('как')
INCLUDING = or_(
    or_(
        eq('в'),
        eq('том'),
        eq('числе'),
    ),
    eq('включающий'),
    or_(
        eq('включающий'),
        eq('в'),
        eq('себя'),
    ),
    or_(
        eq('включающие'),
        eq('в'),
        eq('себя'),
    ),
    eq('включающие'),
    eq('особенно'),

)

MID_S = or_(
    rule(
        or_(
            eq('такой'),
            eq('такие'),
        ),
        eq('как')
    )
)
ATAKJE = rule(
    eq(','),
    eq('а'),
    eq('также')
)

MID = or_(
    rule(
        eq('это')
    ),
    rule(
        eq('—')
    ),
    rule(
        eq('—'),
        eq('это')
    ),
    rule(
        eq('—'),
        not_(eq('км'))
    ),
    rule(
        or_(
            eq('и'),
            eq('или'),
        ),
        eq('другие')
    )
)

END = or_(
    rule(
        gram('NOUN'),
        gram('NOUN')
    ),
    rule(
        gram('ADJF').repeatable(),
        gram('NOUN')
    ),
    rule(
        gram('ADJF'),
        gram('ADJF').repeatable(),
        gram('NOUN')
    ),
    rule(
        gram('NOUN').repeatable(),
        gram('ADJF'),
        gram('NOUN').repeatable()
    ),
    rule(
        gram('NOUN').repeatable()
    )
)

Item = fact(
    'Item',
    [attribute('titles').repeatable()]
)


IGNORE = rule(
    '(',
    not_(eq(')')).repeatable(),
    ')'
)

ITEM = rule(
    IGNORE.interpretation(
        Item.titles
    ),
    eq(',').optional() 
).repeatable().interpretation(
    Item
)


# In[5]:


def get_hyperonyms(main_word):
    HYPONYM = eq(utils.deaccent(main_word))
    RULE = or_(
        rule(HYPONYM, ATAKJE, START, MID, END),
        rule(HYPONYM, MID, END),
        rule(START_S, END, KAK, HYPONYM),
        rule(END, INCLUDING, HYPONYM)
    )
    parser = Parser(RULE) 
    text = utils.deaccent(wikipedia.summary(main_word))
    print(text)
    text = re.sub(r'\(.+?\)', '', text)
    text = text.lower().replace('* сергии радонежскии* ', '')
    for idx, match in enumerate(parser.findall(text.lower())):
        k = [_.value for _ in match.tokens]
        print(k)


# In[6]:


bb_data = pd.read_excel('bb_data.xlsx', sheet_name='hyperonyms')
bb_data.head(10)


# In[8]:


for hyponym in list(bb_data['hyponym']):
    print('Hyperonym for the word: ', hyponym)
    get_hyperonyms(hyponym)
    print('\n')
    
print('Done.')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




