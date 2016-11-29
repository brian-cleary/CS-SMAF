import numpy as np
from dl_simulation import random_phi, get_observations
from analyze_predictions import *
import glob,os
import sys

fp,snr = sys.argv[1:]
snr = float(snr)

iters = 15

prefix = fp[:fp.rfind('/')]
X = np.load(fp)
thresh = np.percentile(X,99.5)
X[(X > thresh)] = thresh
Cluster = []
for _ in range(iters):
	Phi = np.eye(X.shape[0])
	Y = get_observations(X,Phi,snr=snr)
	cluster_similarity = compare_clusters(X,Y)
	Cluster.append(cluster_similarity)

print prefix,np.average(Cluster),np.std(Cluster)