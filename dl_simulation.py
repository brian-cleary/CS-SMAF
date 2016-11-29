import numpy as np
from sklearn import decomposition
from sklearn.linear_model import MultiTaskLassoCV,OrthogonalMatchingPursuit,RidgeCV,Ridge,ElasticNetCV,Lasso
import spams
from scipy.spatial import distance
from scipy.stats import spearmanr, entropy
import sys
from sklearn import mixture

THREADS = 10

def random_phi(m,g,d_thresh=0.2,nonneg=False):
	Phi = np.zeros((m,g))
	Phi[0] = np.random.randn(g)
	if nonneg:
		Phi[0] = abs(Phi[0])
	Phi[0] /= np.linalg.norm(Phi[0])
	for i in range(1,m):
		dmax = 1
		while dmax > d_thresh:
			p = np.random.randn(g)
			if nonneg:
				p = abs(p)
			dmax = max(abs(1 - distance.cdist(Phi,[p],'correlation')))
		Phi[i] = p/np.linalg.norm(p)
	return Phi

def get_observations(X0,Phi,snr=5):
	noise = np.array([np.random.randn(X0.shape[1]) for _ in range(X0.shape[0])])
	noise *= np.linalg.norm(X0)/np.linalg.norm(noise)/snr
	return Phi.dot(X0 + noise)

def sparse_decode(Y,D,k,worstFit=1.,mink=4):
	while k > mink:
		W = spams.omp(np.asfortranarray(Y),np.asfortranarray(D),L=k,numThreads=THREADS)
		W = np.asarray(W.todense())
		fit = 1 - np.linalg.norm(Y - D.dot(W))**2/np.linalg.norm(Y)**2
		if fit < worstFit:
			break
		else:
			k -= 1
	return W

def update_sparse_predictions(Y,D,W,Psi,lda=0.0001):
	X = np.zeros((Psi.shape[0],W.shape[1]))
	for i in range(W.shape[1]):
		used = (W[:,i] != 0)
		if used.sum() > 0:
			d = np.copy(D)
			d = d[:,used]
			model = Ridge(alpha=lda)
			model.fit(d,Y[:,i])
			X[:,i] = model.predict(Psi[:,used])
	return X

def recover_system_knownBasis(X0,m,k,Psi=[],use_ridge=False,snr=0,nsr_pool=0):
	if len(Psi) == 0:
		Psi,s,vt = np.linalg.svd(X0)
	Phi = random_phi(m,X0.shape[0])
	Phi_noise = random_phi(m,X0.shape[0])*nsr_pool
	D = Phi.dot(Psi)
	Y = get_observations(X0,Phi+Phi_noise,snr=snr)
	W = sparse_decode(Y,D,k)
	if use_ridge:
		X = update_sparse_predictions(Y,D,W,Psi)
	else:
		X = Psi.dot(W)
	return X,Phi,Y,W,D,Psi
