import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering, AffinityPropagation

def compare_distances(A,B,random_samples=[],s=200):
	if len(random_samples) == 0:
		random_samples = np.zeros(A.shape[1],dtype=np.bool)
		random_samples[:min(s,A.shape[1])] = True
		np.random.shuffle(random_samples)
	dist_x = distance.pdist(A[:,random_samples].T,'euclidean')
	dist_y = distance.pdist(B[:,random_samples].T,'euclidean')
	p = (1 - distance.correlation(dist_x,dist_y))
	spear = spearmanr(dist_x,dist_y)
	return p,spear[0]

def compare_clusters(X,Y,method='spectral',s=10000):
	A = (X/np.linalg.norm(X,axis=0)).T
	B = (Y/np.linalg.norm(Y,axis=0)).T
	random_samples = np.zeros(A.shape[0],dtype=np.bool)
	random_samples[:min(s,A.shape[0])] = True
	np.random.shuffle(random_samples)
	A = A[random_samples]
	B = B[random_samples]
	dA = 1 - A.dot(A.T)
	dA = np.exp(-dA**2/2.)
	dB = 1 - B.dot(B.T)
	dB = np.exp(-dB**2/2.)
	del A,B
	if method == 'spectral':
		n = max(5,min(30,X.shape[1]/50))
		lA = SpectralClustering(n_clusters=n,affinity='precomputed').fit_predict(dA)
		lB = SpectralClustering(n_clusters=n,affinity='precomputed').fit_predict(dB)
	elif method == 'ap':
		lA = AffinityPropagation(affinity='precomputed').fit_predict(dA)
		lB = AffinityPropagation(affinity='precomputed').fit_predict(dB)
	return adjusted_mutual_info_score(lA,lB)

def correlations(A,B,pc_n=100):
	p = (1 - distance.correlation(A.flatten(),B.flatten()))
	spear = spearmanr(A.flatten(),B.flatten())
	dist_genes = np.zeros(A.shape[0])
	for i in range(A.shape[0]):
		dist_genes[i] = 1 - distance.correlation(A[i],B[i])
	pg = (np.average(dist_genes[np.isfinite(dist_genes)]))
	dist_sample = np.zeros(A.shape[1])
	for i in range(A.shape[1]):
		dist_sample[i] = 1 - distance.correlation(A[:,i],B[:,i])
	ps = (np.average(dist_sample[np.isfinite(dist_sample)]))
	pc_dist = []
	if pc_n > 0:
		u0,s0,vt0 = np.linalg.svd(A)
		u,s,vt = np.linalg.svd(B)
		for i in range(pc_n):
			pc_dist.append(abs(1 - distance.cosine(u0[:,i],u[:,i])))
		pc_dist = np.array(pc_dist)
	return p,spear[0],pg,ps,pc_dist
