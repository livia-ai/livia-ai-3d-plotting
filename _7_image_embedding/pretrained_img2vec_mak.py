import csv
import glob
import numpy as np
import pandas as pd
from progress.bar import Bar
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.decomposition import PCA

# https://github.com/christiansafka/img2vec

# EfficientNet-B3 produces vectors with 1536 dimensions 
img2vec = Img2Vec(cuda=False, model='efficientnet_b3')

vectors = []

# MAK has a correspondence file to get record ID <-> image filename,
# which we'll loop through
# open file in read mode
records_csv = '/home/rainers/Workspaces/livia/data/MAK/mak_metadata_images_bewilligt.csv'

with open(records_csv, 'r') as infile:
  reader = csv.reader(infile)

  next(reader, None)  # skip header

  # bar = Bar('Processing', max=len(filenames))

  ctr = 1

  for row in reader:
    priref = row[0]
    
    url = row[4]

    filepath = '/home/rainers/Workspaces/livia/data/MAK/images' + url[url.rfind('/publikationsbilder'):]

    try:
      img = Image.open(filepath)
      img = img.convert('RGB') # this sucks 

      vec = img2vec.get_vec(img)

      result_row = [ priref ]
      result_row.extend(vec)

      vectors.append(result_row)
    except Exception as e:
      print('Error loading image: ' + filepath)

    print('Row ' + str(ctr), end='\r')
    ctr += 1

print('writing results to file')

with open('results.csv', 'w') as outfile:
  writer = csv.writer(outfile)
  writer.writerows(vectors)

print('reducing dimensions')

pca = PCA(n_components=128, svd_solver='full')
vec_reduced = pca.fit_transform(np.array([ row[1:] for row in vectors ]))
vec_reduced = [ row.tolist() for row in vec_reduced ]

print('writing results to file')

for idx, vec in enumerate(vec_reduced):
  id = vectors[idx][0]
  vec_reduced[idx].insert(0, id)

with open('results_128.csv', 'w') as outfile:  
  writer = csv.writer(outfile)
  writer.writerows(vec_reduced)

print('done')
