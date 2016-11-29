import numpy as np
from dl_simulation import random_phi, get_observations
from union_of_transforms import random_submatrix
from analyze_predictions import *
from scipy.spatial import distance
from scipy.stats import spearmanr
import glob,os
import sys

fp,snr = sys.argv[1:]
snr = float(snr)

iters = 10
M = [10,25,50,100,200,400]

prefix = fp[:fp.rfind('/')]
X = np.load(fp)
thresh = np.percentile(X,99.5)
X[(X > thresh)] = thresh
for m in M:
	Cluster = []
	for _ in range(iters):
		Phi = random_phi(m,X.shape[0])
		Y = get_observations(X,Phi,snr=snr)
		pearson_dist,spearman_dist = compare_distances(X,Y)
		cluster_similarity = compare_clusters(X,Y,method='ap',s=500)
		Cluster.append(cluster_similarity)
	print prefix,m,np.average(Cluster)