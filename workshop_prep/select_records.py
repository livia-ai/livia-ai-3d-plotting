"""
Select records from the full dataset, based on a list of IDs
"""

import csv
import pandas as pd

# 'Good' example
# DATASET = "MAK"
# KEY = "priref"
# ID = "239530"
# NEIGHBOURS = [
#   "239532", "239531", "239523", "239522", "239534", "239524", "239519", "239545", "239543", "239527", "239528", "239553", "239566", "239525", "239427", "239567", "239526", "239529", "239551", "239463", "239533", "239480", "239478", "239479", "239541", "239610", "239462", "239550", "239554", "239512"
# ]
# METADATA = pd.read_csv("../data/mak_flattened_all.csv")

# 'Less good' example (they're all good, really)
# DATASET = "MAK"
# KEY = "priref"
# ID = "178255"
# NEIGHBOURS = [
#   "177338", "177340", "177344", "177342", "177820", "177461", "177811", "176816", "176767", "177462", "177542", "178387", "177532", "177810", "177818", "177515", "177204", "177405", "177924", "177419", "177503", "177805", "177806", "177804", "177420", "176446", "177100", "177101", "178685", "177923"
# ]
# METADATA = pd.read_csv("../data/mak_flattened_all.csv")

# Belvedere good
# DATASET = "BEL"
# KEY = "Identifier"
# ID = "751"
# NEIGHBOURS = [
#   "1665", "4785", "7940", "517", "727", "3789", "1758", "1894", "904", "6180", "1164", "5802", "1124", "1414", "1451", "1023", "7810", "8613", "4384", "433g", "6589", "939", "343", "1131", "1824", "880", "413a", "6594", "1065", "9255"
# ]
# METADATA = pd.read_csv("../data/belvedere_flattenend_20220315.csv")

# Belvedere not so good
# DATASET = "BEL"
# KEY = "Identifier"
# ID = "4242"
# NEIGHBOURS = [
#   "3380", "4198", "6642", "3196", "2376", "4048", "2086", "4298", "8500", "2427", "2097", "1486", "1513", "4073", "8501", "4828", "4438", "4303", "3171", "494", "5987/5", "4402", "2244", "11786", "1803", "4238", "8507", "1768", "4430", "4244"
# ]
# METADATA = pd.read_csv("../data/belvedere_flattenend_20220315.csv")

# Wien Museum good
# DATASET = "WM"
# KEY = "id"
# ID = "676744"
# NEIGHBOURS = [ 
#   "676730", "686082", "1059327", "681112", "676861", "984170", "676930", "668574", "668575", "677516", "686019", "676807", "676753", "668547", "676917", "686176", "686175", "685894", "676915", "685798", "685799", "685898", "685895", "677192", "677191", "685794", "1028452", "693111", "693136", "686157"
# ]
# METADATA = pd.read_csv("../data/wien_museum.csv")

# Wien Museum not so good
# DATASET = "WM"
# KEY = "id"
# ID = "95201"
# NEIGHBOURS = [
#   "92103", "89405", "93424", "96660", "101246", "93380", "91795", "46989", "95204", "91516", "94963", "93430", "91477", "93413", "93039", "102966", "181002", "91522", "92844", "440960", "93404", "91480", "90315", "180993", "181484", "86972", "311617", "91513", "91066", "94972"
# ]
# METADATA = pd.read_csv("../data/wien_museum.csv")

###
# Triplet examples
###

## Wien Museum
# DATASET = "WM"
# KEY = "id"
# ID = "1040089", 
# NEIGHBOURS = [
#   "1040089", "1040091", "809581", "190349", "221078", "390050", "549629", "548803", "77823", "656238", "655117", "180783", "677108", "677087", "426511"
# ]
# METADATA = pd.read_csv("../data/wien_museum.csv")

# Belvedere
# DATASET = "BEL"
# KEY = "Identifier"
# ID = "BB_6274-072"
# NEIGHBOURS = [
#   "BB_6274-072", "BB_6274-092", "7886", "46", "5929", "5692b", "2098", "4653", "4914", "6257", "6255", "7882", "4972", "4876", "7196"
# ]
# METADATA = pd.read_csv("../data/belvedere_flattenend_20220315.csv")

# MAK
DATASET = "MAK"
KEY = "priref"
ID = "262598"
NEIGHBOURS = [
  "262598", "262596", "50246", "247094", "245037", "316005", "277749", "277748", "29519", "316004", "336180", "129012", "46105", "46112", "233223"
]
METADATA = pd.read_csv("../data/mak_flattened_all.csv")

result_samples = []
result_samples.append(list(METADATA.columns.values))

NEIGHBOURS.insert(0, ID)

for id in NEIGHBOURS:
  ref = id 

  if DATASET == "WM" or DATASET == "MAK":
    ref = float(id)

  record = METADATA.loc[METADATA[KEY] == ref].values.flatten().tolist()
  result_samples.append(record) 
  
with open('output.csv', 'w') as out:
  writer = csv.writer(out)
  writer.writerows(result_samples)