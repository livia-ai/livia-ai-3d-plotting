import csv
from re import M
import pandas as pd

EMBEDDINGS_FILE = "../data/se_wm_100d.csv"

metadata = pd.read_csv("../data/wien_museum.csv")

merged = []

with open(EMBEDDINGS_FILE, 'r') as f:
  reader = csv.reader(f)

  for row in reader:
    identifier = row[0]

    record = metadata.loc[metadata['id'] == float(identifier)]
    relevant_fields = record[['id', 'url', 'multimedia_default']].values.flatten().tolist()

    merged.append(relevant_fields + row[1:])

with open('output.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerows(merged)

