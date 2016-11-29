import numpy as np
from sklearn.linear_model import MultiTaskLasso,RidgeCV
from scipy.spatial import distance
from scipy.stats import spearmanr
from union_of_transforms import random_submatrix
from dl_simulation import spectral_compare

def get_signature_genes(X,n,lda=10):
	W = np.zeros((X.shape[0],X.shape[0]))
	# coarse search from the bottom
	while (abs(W).sum(1) > 0).sum() < n:
		lda /= 10.
		model = MultiTaskLasso(alpha=lda,max_iter=100,tol=.001,selection='random',warm_start=True)
		model.fit(X.T,X.T)
		W = model.coef_.T
		#print len(np.nonzero(abs(W).sum(1))[0]),model.score(X.T,X.T)
	# fine search from the top
	while (abs(W).sum(1) > 0).sum() > n*1.2:
		lda *= 2.
		model.set_params(alpha=lda)
		model.fit(X.T,X.T)
		W = model.coef_.T
		#print len(np.nonzero(abs(W).sum(1))[0]),model.score(X.T,X.T)
	# finer search
	while (abs(W).sum(1) > 0).sum() > n:
		lda *= 1.1
		model.set_params(alpha=lda)
		model.fit(X.T,X.T)
		W = model.coef_.T
		#print len(np.nonzero(abs(W).sum(1))[0]),model.score(X.T,X.T)
	return np.nonzero(abs(W).sum(1))[0]

def build_signature_model(X,gidx,n_alphas=5):
	model = RidgeCV(alphas=(.1,1,10,100,1000,10000,100000),cv=5)
	model.fit(X[gidx].T,X.T)
	return model

def signature_predictions(X0,n_test_samples,n_signatures):
	sidx = np.zeros(X0.shape[1],dtype=np.bool)
	sidx[:n_test_samples] = True
	np.random.shuffle(sidx)
	Xtrain = X0[:,np.invert(sidx)]
	Xtest = X0[:,sidx]
	sg = get_signature_genes(Xtrain,n_signatures,lda=10000000)
	model = build_signature_model(Xtrain,sg)
	Xhat = (model.predict(Xtest[sg].T)).T
	return sg,Xtest,Xhat
