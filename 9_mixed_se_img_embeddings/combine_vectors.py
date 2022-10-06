import csv
import numpy as np
from sklearn.decomposition import PCA

# Load 256d sentence embeddings
# se_csv = '../data/sentence_embeddings/bel_sbert_Title_Description_ExpertTags_256d.csv'
# se_csv = '../data/sentence_embeddings/wm_sbert_title_subjects_256d.csv'
se_csv = '../data/sentence_embeddings/mak_sbert_title_description_256d.csv'

se = []

print('Loading embeddings')

with open(se_csv, 'r') as infile:
  reader = csv.reader(infile)

  for row in reader:
    se.append(row)

# Load 128d image embeddings
# ie_csv = '../../data/Belvedere/bel_image_embedding.csv'
# ie_csv = '../../data/Wien_Museum/wm_image_embedding.csv'
ie_csv = '../../data/MAK/mak_image_embedding_priref.csv'

ie = {}

with open(ie_csv, 'r') as infile:
  reader = csv.reader(infile)

  for row in reader:
    id = row[0].replace('__@@__', '/')
    vec = row[1:]
    ie[id] = vec

print('Reducing sentence embeddings to 128d')

# Run PCA on sentence embeddings, to reduce to 128d
pca = PCA(n_components=128, svd_solver='full')
se_reduced = pca.fit_transform(np.array([ row[1:] for row in se ]))
se_reduced = [ row.tolist() for row in se_reduced ]

print('Concatenating vectors')

combined = []

for idx, se_128 in enumerate(se_reduced):
  id = se[idx][0]

  if id in ie:
    ie_128 = ie[id]

    c = [ id ] + ie_128 + se_128

    combined.append(c)
  else:
    print('No image vector for ' + id)

print('Writing results to file')

with open('combined_embeddings_256d.csv', 'w') as outfile:  
  writer = csv.writer(outfile)
  writer.writerows(combined)