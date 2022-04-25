import csv
from re import M
import pandas as pd

EMBEDDINGS_FILE = "../data/se_mak_100d.csv"

metadata = pd.read_csv("../data/mak_flattened_all.csv")

merged = []

with open(EMBEDDINGS_FILE, 'r') as f:
  reader = csv.reader(f)

  for row in reader:
    priref = row[0]

    record = metadata.loc[metadata['priref'] == float(priref)]
    relevant_fields = record[['priref', 'filename']].values.flatten().tolist()

    relevant_fields_normalized = [
      relevant_fields[0],
      'https://sammlung.mak.at/sammlung_online?id=' + relevant_fields[1][:-4],
      ''
    ]

    merged.append(relevant_fields_normalized + row[1:])

with open('output.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerows(merged)

