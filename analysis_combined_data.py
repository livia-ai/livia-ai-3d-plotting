import numpy as np
import pandas as pd
from utility_functions import utility_functions as utils
# load combined dataframe
df_combined = np.loadtxt('data/combined_dataset_100d_numpy.csv', delimiter=',',dtype=object)
df_combined = pd.DataFrame(df_combined, columns=["museum", "id", "url", "media_url"] + list(range(100)))
#print(df_combined)

# load different datasets
wm_data = pd.read_csv("data/wien_museum.csv")
# mak
mak_1 = pd.read_csv("data/mak_1.csv", low_memory=False)
mak_2 = pd.read_csv("data/mak_2.csv", low_memory=False)
mak_3 = pd.read_csv("data/mak_3.csv", low_memory=False )
mak = pd.concat([mak_1, mak_2, mak_3])
# belvedere
bel = pd.read_csv("data/belvedere.csv").reset_index()

## extract interesting columns
wm_filtered = wm_data[wm_data.columns[[0,3,4,5,6,7,8]]]
mak_filtered = mak[mak.columns[[0,2,4,5,15,16,17,21,28,29,31,34,36,38]]]
mak_filtered.reset_index(drop=True, inplace=True)
bel_filtered = bel[bel.columns[[0,10,1,2,3,4,11,12,14,16,21,22]]]

# create one column that contains all text data
#wm_filtered = wm_filtered.assign(full_text = wm_filtered[wm_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
#mak_filtered = mak_filtered.assign(full_text = mak_filtered[mak_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))
#bel_filtered = bel_filtered.assign(full_text = bel_filtered[bel_filtered.columns[2:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))

embedding = df_combined[list(range(100))].to_numpy(dtype=np.float64)

# specify id manually
#museum = "wm"
#id = 930
#sample = df_combined.loc[(df_combined["museum"] == museum) & (df_combined["id"] == str(id) )]
#query_id = np.array(sample.index)
#query = sample[list(range(100))].to_numpy(dtype=np.float64)

# randomly sample ids
n = 200
rng = np.random.default_rng()
id_list = list(range(len(embedding)))
query_ids = rng.choice(id_list, size=n, replace=False)
queries = embedding[query_ids]

full_information, triplets = utils.create_triplets(queries, query_ids, embedding, "cosine", 100)

# analize results
for i in range(len(full_information)):
    query_id = query_ids[i]
    min_k_ids, min_k_dists, max_k_ids, max_k_dists = full_information[i]
    
    sim_df = df_combined.loc[min_k_ids]
    query_df = df_combined.loc[query_id]

    museum_set = set(sim_df["museum"])
    if len(museum_set) > 1:

        sample_m = query_df["museum"]
        sample_id = query_df["id"]
        print(f"sample from {sample_m}")
        if sample_m == "wm":
            print(wm_filtered.loc[wm_filtered["id"] == int(sample_id)]["title"])

        if sample_m == "mak":
            print(mak_filtered.loc[mak_filtered["priref"] == int(sample_id)]["title"])

        if sample_m == "bel":
            print(bel_filtered.loc[bel_filtered["Identifier"] == sample_id]["Title"])

        counter = 1
        print("similar samples from different museums")
        for i in sim_df[["id", "museum"]].itertuples():
            m = i[2]
            idx = i[1]
            if m != sample_m:
                print(counter, m)
                if m == "wm":
                    print(wm_filtered.loc[wm_filtered["id"] == int(idx)]["title"])

                if m == "mak":
                    print(mak_filtered.loc[mak_filtered["priref"] == int(idx)]["title"])

                if m == "bel":
                    print(bel_filtered.loc[bel_filtered["Identifier"] == idx]["Title"])
            counter += 1
        print()
        print()