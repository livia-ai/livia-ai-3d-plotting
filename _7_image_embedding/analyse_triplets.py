 
import pickle
from collections import defaultdict
import livia.triplet as triplet
import livia.embedding as embedding
from dataset import TripletDataset

#############################
## generate triplets
#n = 5000
#embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
#triplets = triplet.generate_triplets(embedding_loaded, "clustering", n)
#with open(f'data/bel_triplets_anlysis_nn=50', 'wb') as fp:
#    pickle.dump(triplets, fp)
#############################


############################
# load triplets
with open(f'data/bel_image_paths', 'rb') as fp:
    triplets = pickle.load(fp)
############################


##########################
# analyse triplets

sim_counts = defaultdict(int)
dis_counts = defaultdict(int)
sam_counts = defaultdict(int)
for sample, sim, dis in triplets:
    sam_counts[sample] += 1
    sim_counts[sim] += 1
    dis_counts[dis] += 1

n = len(triplets)
k = 100
max_k_dis = sorted(dis_counts.items(), key=lambda x: x[1], reverse=True)[:k]
max_k_sim = sorted(sim_counts.items(), key=lambda x: x[1], reverse=True)[:k]
max_k_ori = sorted(sam_counts.items(), key=lambda x: x[1], reverse=True)[:k]

with open("new_triplets_analysis.txt", "w") as f:
    f.write(str(max_k_dis) + "\n\n")
    f.write(str(max_k_sim) + "\n\n")
    f.write(str(max_k_ori) + "\n\n")

sum_ori = 0
sum_dis = 0
sum_sim = 0
for i in range(k):
    sum_dis += max_k_dis[i][1]
    sum_sim += max_k_sim[i][1]
    sum_ori += max_k_ori[i][1]


print("dis:", sum_dis/n)
print("sim:", sum_sim/n)
print("ori:", sum_ori/n)


#######################################
## plot some triplets
#useable_triplets = []
#for ori, sim, dis in triplets:

#    split_ori = ori.split("/")
#    split_sim = sim.split("/")
#    split_dis = dis.split("/")

#    useable_triplets.append(("__@@__".join(split_ori), "__@@__".join(split_sim), "__@@__".join(split_dis)))


#root_dir = "data/images/bel_cropped"
#dataset = TripletDataset(triplets = useable_triplets,
#                         root_dir = root_dir)

#import matplotlib.pyplot as plt
#for triplet in dataset: 

#    print(triplet[3])

#    fig,ax = plt.subplots(1,3)
#    ax[0].imshow(triplet[0])
#    ax[1].imshow(triplet[1])
#    ax[2].imshow(triplet[2])

#    ax[0].set_title("Sample")
#    ax[1].set_title("Similar")
#    ax[2].set_title("Dissimilar")
#    plt.show()


