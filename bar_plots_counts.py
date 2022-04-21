#imports
import numpy as np
import pandas as pd
import utility_functions.utility_functions as utils
from sklearn.decomposition import PCA


# data loading and preprocessing
dataset_name = "BEL"

if dataset_name == "WM":
    # load and prepare wien museum
    wm_original = pd.read_csv("data/wien_museum.csv") 
    wm_filtered = wm_original[wm_original.columns[[0,3,4,5,6,7,8]]]
    df = wm_filtered
    # dataset specific column name used for coloring
    column_to_count = "classifications" # column used for coloring
    doc_name = f"wm_counts_{column_to_count}"

if dataset_name == "MAK":
    # load and prepare mak
    mak_1 = pd.read_csv("data/mak_1.csv")
    mak_2 = pd.read_csv("data/mak_2.csv")
    mak_3 = pd.read_csv("data/mak_3.csv")
    mak = pd.concat([mak_1, mak_2, mak_3])
    mak_filtered = mak[mak.columns[[0,2,4,5,15,16,17,21,28,29,31,34,36,38]]]
    mak_filtered.reset_index(drop=True, inplace=True)
    df = mak_filtered
    # dataset specific column name used for coloring
    column_to_count = "collection" # "collection", "object_name"
    doc_name = f"mak_counts_{column_to_count}"

if dataset_name == "BEL":
    # load and prepare belvedere
    bel = pd.read_csv("data/belvedere.csv").reset_index()
    bel_filtered = bel[bel.columns[[0,10,1,2,3,4,11,12,14,16,21,22]]]
    df = bel_filtered
    # dataset specific column name used for coloring
    column_to_count = "ObjectClass" # ObjectClass, Collection
    doc_name = f"bel_counts_{column_to_count}"


counts = df[column_to_count].value_counts().to_frame(name = "counts")
counts.index.name = column_to_count
counts.reset_index(inplace=True)

# filter all classes that have counts smaller than some threshold
threshold = 5
counts_filtered = counts[counts["counts"] > threshold]

# Some statistics about the filtering saved in a text file
perc = round(sum(counts_filtered["counts"])/sum(counts["counts"]) *100, 2)
remaining = sum(counts_filtered["counts"])
discarded = sum(counts["counts"]) - sum(counts_filtered["counts"])
with open("blog_visualizations/reading_meta_data/" + doc_name + ".txt" , "w") as text_file:
    text_file.write('Before creating the bar plot the counts have been filtered!\n' +\
                    f'Cutoff value: {threshold} \n\n' +\
                    f"Classes remaining: {len(counts_filtered)}\n" +\
                    f"Classes discarded: {len(counts) - len(counts_filtered)}\n\n" +\
                    f"Remaining samples: {remaining}\n" +\
                    f"Discarded samples: {discarded}\n" +\
                    f"Remaining samples in percent: {perc}%\n" +\
                    f"Discarded samples in percent: {round(100-perc,2)}%\n")

# create bar plots
utils.plot_counts_full(counts_filtered, "blog_visualizations/reading_meta_data/" + doc_name, column_to_count)