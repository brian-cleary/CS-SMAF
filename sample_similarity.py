import numpy as np
from dl_simulation import random_phi, get_observations
from union_of_transforms import random_submatrix
from analyze_predictions import *
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.manifold import Isomap,MDS
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
	Pearson = []
	Spearman = []
	Pearson_p = []
	Spearman_p = []
	Pearson_MDS = []
	Pearson_Iso = []
	Pearson_MDS_p = []
	Pearson_Iso_p = []
	Pearson_svd = []
	Spearman_svd = []
	Cluster = []
	Cluster_svd = []
	for _ in range(iters):
		Phi = random_phi(m,X.shape[0])
		Y,noise = get_observations(X,Phi,snr=snr,return_noise=True)
		pearson_dist,spearman_dist = compare_distances(X,Y,pvalues=True)
		cluster_similarity = compare_clusters(X,Y)
		Pearson.append(pearson_dist[0])
		Spearman.append(spearman_dist[0])
		Pearson_p.append(pearson_dist[1])
		Spearman_p.append(spearman_dist[1])
		Cluster.append(cluster_similarity)
		X_mds = MDS().fit_transform(X.T).T
		Y_mds = MDS().fit_transform(Y.T).T
		X_iso = Isomap().fit_transform(X.T).T
		Y_iso = Isomap().fit_transform(Y.T).T
		pearson_mds,spearman_mds = compare_distances(X_mds,Y_mds,pvalues=True)
		pearson_iso,spearman_iso = compare_distances(X_iso,Y_iso,pvalues=True)
		Pearson_MDS.append(pearson_mds[0])
		Pearson_Iso.append(pearson_iso[0])
		Pearson_MDS_p.append(pearson_mds[1])
		Pearson_Iso_p.append(pearson_mds[1])
		ua,sa,vta = np.linalg.svd(X+noise,full_matrices=False)
		Vt = np.diag(sa).dot(vta)
		pearson_svd,spearman_svd = compare_distances(Vt[:m],Y,pvalues=False)
		cluster_similarity_svd = compare_clusters(Vt[:m],Y)
		Pearson_svd.append(pearson_svd)
		Spearman_svd.append(spearman_svd)
		Cluster_svd.append(cluster_similarity_svd)
	print prefix,m,np.average(Pearson),np.average(Pearson_p),np.average(Spearman),np.average(Spearman_p),np.average(Pearson_MDS),np.average(Pearson_MDS_p),np.average(Pearson_Iso),np.average(Pearson_Iso_p),np.average(Cluster),np.average(Pearson_svd),np.average(Spearman_svd),np.average(Cluster_svd)
