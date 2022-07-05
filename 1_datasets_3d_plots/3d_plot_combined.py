# imports
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from utility_functions import utility_functions as utils

#### load data ####
# wien museum
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


#### bring different dataframes into the same format ####
# columns: ["id", "id_column_name", "title", "museum"] 
# "id" -> identifier of the sample
# "id_column_name" -> how the identifier column is called in the original dataframe
# "title" -> title of sample
# "museum" -> abbreviation of the museum: wm, mak or bel

# wm
wm_copy = wm_filtered.copy()
wm_copy = wm_copy.assign(id_column_name = "id")
wm_copy = wm_copy.assign(museum = "wm")
wm_copy = wm_copy.assign(url=wm_data["url"])
wm_copy = wm_copy.assign(media_url=wm_data["multimedia_default"])
wm_to_combine = wm_copy[["id","id_column_name", "title", "museum" ]]
# to create a combined dataset
wm_for_dataset = wm_copy[["museum", "id", "url", "media_url"]]
#print(wm_to_combine.head())

# mak
mak_copy = mak_filtered.copy()
mak_copy = mak_copy.assign(id_column_name = "priref")
mak_copy = mak_copy.assign(id = mak_copy["priref"])
mak_copy = mak_copy.assign(museum = "mak")
mak_copy = mak_copy.assign(url=mak_copy["priref"].apply(lambda x: f"https://sammlung.mak.at/sammlung_online?id=collect-{x}"))

mak_to_combine = mak_copy[["id","id_column_name", "title", "museum" ]]
# to create a combined dataset
mak_for_dataset = mak_copy[["museum", "id", "url"]]
#print(mak_to_combine.head())

#bel
bel_copy = bel_filtered.copy()
bel_copy = bel_copy.assign(id_column_name = "Identifier")
bel_copy = bel_copy.assign(id = bel_filtered["Identifier"])
bel_copy = bel_copy.assign(title = bel_filtered["Title"])
bel_copy = bel_copy.assign(museum = "bel")
bel_copy = bel_copy.assign(url=bel["IsShownAt"])
bel_copy = bel_copy.assign(media_url=bel["Object"])

bel_to_combine = bel_copy[["id","id_column_name", "title", "museum" ]]
# to create a combined dataset
bel_for_dataset = bel_copy[["museum", "id", "url", "media_url"]]
#print(bel_to_combine.head())

#### combine dataframes ####
df_combined_plot = pd.concat([wm_to_combine, mak_to_combine, bel_to_combine]).reset_index(drop=True)
#df_combined_dataset = pd.concat([wm_for_dataset, mak_for_dataset, bel_for_dataset]).reset_index(drop=True)
#print(df_combined_plot)
#print(df_combined_dataset)

#### load and downproject sentence_embedding to 3d ####
se_combined = np.loadtxt('data/combined_dataset_100d_numpy.csv', delimiter=',', usecols=range(4,104))

n_components = 3
pca = PCA(n_components=n_components)
dp_embeddings = pca.fit_transform(se_combined)


#### plot ####
# sample n
for n in [100,500,1000,2500]:
    rng = np.random.default_rng()
    id_list = list(range(len(dp_embeddings)))
    sample_ids = rng.choice(id_list, size=n, replace=False)
    samples = dp_embeddings[sample_ids]
    print(len(set(sample_ids)))

    ## plot full dataset
    #sample_ids = np.array(list(range(len(dp_embeddings))))
    #samples = dp_embeddings[:]

    # plot results 
    title = f"Combined Dataset, {n} Random Samples"
    utils.plot(meta_data = df_combined_plot, 
            sample_ids = sample_ids,
            embeddings = samples,  
            color = "museum",
            identifier = "id",
            hover_name = "title", 
            title = title)


