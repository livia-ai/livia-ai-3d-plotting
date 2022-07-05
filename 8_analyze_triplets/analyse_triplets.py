 
import pickle
from collections import defaultdict
import livia.triplet as triplet
import livia.embedding as embedding


############################
## generate triplets
#n = 10000
#embedding_loaded = embedding.load_csv("data/bel/bel_sbert_Title_Description_ExpertTags_512d.csv")
#triplets = triplet.generate_triplets(embedding_loaded, "clustering", n)
#with open(f'8_analyze_triplets/bel_triplets_anlysis_1', 'wb') as fp:
#    pickle.dump(triplets, fp)
############################


############################
## load triplets
#with open(f'8_analyze_triplets/bel_triplets_anlysis_1', 'rb') as fp:
#    triplets = pickle.load(fp)
############################


###########################
## analyse triplets

#sim_counts = defaultdict(int)
#dis_counts = defaultdict(int)
#sam_counts = defaultdict(int)
#for sample, sim, dis in triplets:
#    sam_counts[sample] += 1
#    sim_counts[sim] += 1
#    dis_counts[dis] += 1

#n = len(triplets)
#k = 100
#max_k_dis = sorted(dis_counts.items(), key=lambda x: x[1], reverse=True)[:k]
#max_k_sim = sorted(sim_counts.items(), key=lambda x: x[1], reverse=True)[:k]
#max_k_ori = sorted(sam_counts.items(), key=lambda x: x[1], reverse=True)[:k]

#sum_ori = 0
#sum_dis = 0
#sum_sim = 0
#for i in range(k):
#    sum_dis += max_k_dis[i][1]
#    sum_sim += max_k_sim[i][1]
#    sum_ori += max_k_ori[i][1]

#print("dis:", sum_dis/n)
#print("sim:", sum_sim/n)
#print("ori:", sum_ori/n)