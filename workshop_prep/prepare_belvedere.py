import csv
from re import M
import pandas as pd

EMBEDDINGS_FILE = "../data/se_bel_100d.csv"

metadata = pd.read_csv("../data/belvedere_flattenend_20220315.csv")

merged = []

with open(EMBEDDINGS_FILE, 'r') as f:
  reader = csv.reader(f)

  for row in reader:
    identifier = row[0]

    record = metadata.loc[metadata['Identifier'] == identifier]
    relevant_fields = record[['Identifier', 'IsShownAt', 'Object']].values.flatten().tolist()

    merged.append(relevant_fields + row[1:])

with open('output.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerows(merged)

