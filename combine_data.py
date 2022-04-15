from distutils.archive_util import make_archive
import pandas as pd
import numpy as np
import utility_functions.utility_functions as utils
from sentence_transformers import SentenceTransformer


## wien museum
#wm_data = pd.read_csv("data/wien_museum.csv")
## mak
#mak_1 = pd.read_csv("data/mak_1.csv")
#mak_2 = pd.read_csv("data/mak_2.csv")
#mak_3 = pd.read_csv("data/mak_3.csv")
#mak = pd.concat([mak_1, mak_2, mak_3])
## belvedere
#bel = pd.read_csv("data/belvedere.csv").reset_index()

## extract interesting columns
#wm_filtered = wm_data[wm_data.columns[[0,3,4,5,6,7,8]]]
#mak_filtered = mak[mak.columns[[0,2,4,15,16,17,21,28,29,31,34,36,38]]]
#mak_filtered.reset_index(drop=True, inplace=True)
#bel_filtered = bel[bel.columns[[0,1,2,3,10,11,13,15,21]]]

## create one column that contains all text data
#wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
#mak_filtered = mak_filtered.assign(full_text = mak_filtered[mak_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
#bel_filtered = bel_filtered.assign(full_text = bel_filtered[bel_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
#print("Filtering done")

## apply preprocessing
#wm_preprocessed = utils.preprocessing(wm_filtered, "full_text")
#mak_preprocessed = utils.preprocessing(mak_filtered, "full_text")
#bel_preprocessed = utils.preprocessing(bel_filtered, "full_text")
#print("Preprocessing done")

## create 
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

#text_list = []

#for text in wm_preprocessed["pre_text"]:
#    text_list.append(' '.join(text))
#for text in mak_preprocessed["pre_text"]:
#    text_list.append(' '.join(text))
#for text in bel_preprocessed["pre_text"]:
#    text_list.append(' '.join(text))

#sentence_embeddings_combined = model.encode(text_list)
#print(sentence_embeddings_combined.shape)

#np.savetxt('sentence_embeddings_combined.csv', sentence_embeddings_combined, delimiter=',')

combined = np.loadtxt('sentence_embeddings_combined.csv', delimiter=',')
wm = np.loadtxt('data/sentence_embeddings_wien_museum.csv', delimiter=',')
mak = np.loadtxt('data/sentence_embeddings_mak.csv', delimiter=',')
bel = np.loadtxt('data/sentence_embeddings_bel.csv', delimiter=',')

concat = np.concatenate([wm, mak, bel], axis=0)
print(np.sum(combined - concat))