import numpy as np
from dl_simulation import *
from analyze_predictions import *
import sys

THREADS=5

def smaf(X,d,lda1,lda2,maxItr=10,UW=None,posW=False,posU=True,use_chol=False,module_lower=500,activity_lower=5,donorm=False,mode=1,mink=5,U0=[],U0_delta=0.1,doprint=False):
	# use Cholesky when we expect a very sparse result
	# this tends to happen more on the full vs subsampled matrices
	if UW == None:
		U,W = spams.nmf(np.asfortranarray(X),return_lasso=True,K = d,numThreads=THREADS)
		W = np.asarray(W.todense())
	else:
		U,W = UW
	Xhat = U.dot(W)
	Xnorm = np.linalg.norm(X)**2/X.shape[1]
	for itr in range(maxItr):
		if mode == 1:
			# In this mode the ldas correspond to an approximate desired fit
			# Higher lda will be a worse fit, but will result in a sparser sol'n
			U = spams.lasso(np.asfortranarray(X.T),D=np.asfortranarray(W.T),
			lambda1=lda2*Xnorm,mode=1,numThreads=THREADS,cholesky=use_chol,pos=posU)
			U = np.asarray(U.todense()).T
		elif mode == 2:
			if len(U0) > 0:
				U = projected_grad_desc(W.T,X.T,U.T,U0.T,lda2,U0_delta,maxItr=400)
				U = U.T
			else:
				U = spams.lasso(np.asfortranarray(X.T),D=np.asfortranarray(W.T),
				lambda1=lda2,lambda2=0.0,mode=2,numThreads=THREADS,cholesky=use_chol,pos=posU)
				U = np.asarray(U.todense()).T
		if donorm:
			U = U/np.linalg.norm(U,axis=0)
			U[np.isnan(U)] = 0
		if mode == 1:
			wf = (1 - lda2)
			W = sparse_decode(X,U,lda1,worstFit=wf,mink=mink)
		elif mode == 2:
			if len(U0) > 0:
				W = projected_grad_desc(U,X,W,[],lda1,0.,nonneg=posW,maxItr=400)
			else:
				W = spams.lasso(np.asfortranarray(X),D=np.asfortranarray(U),
				lambda1=lda1,lambda2=1.0,mode=2,numThreads=THREADS,cholesky=use_chol,pos=posW)
				W = np.asarray(W.todense())
		Xhat = U.dot(W)
		module_size = np.average([np.exp(entropy(u)) for u in U.T if u.sum()>0])
		activity_size = np.average([np.exp(entropy(abs(w))) for w in W.T])
		if doprint:
			print distance.correlation(X.flatten(),Xhat.flatten()),module_size,activity_size,lda1,lda2
		if module_size < module_lower:
			lda2 /= 2.
		if activity_size < activity_lower:
			lda2 /= 2.
	return U,W

def random_submatrix(Z,g,s,o,threshpct=99,returnIdx=False):
	xg = np.zeros(Z.shape[0],dtype=np.bool)
	xg[:g] = True
	np.random.shuffle(xg)
	xt = np.zeros(Z.shape[1],dtype=np.bool)
	xt[:s] = True
	np.random.shuffle(xt)
	X0 = Z[xg][:,xt]
	thresh = np.percentile(X0,threshpct)
	X0[(X0 > thresh)] = thresh
	xo = np.zeros(X0.shape[0],dtype=np.bool)
	xo[:o] = True
	np.random.shuffle(xo)
	Xobs = X0[xo]
	if returnIdx:
		return X0,xo,Xobs,xt,xg
	else:
		return X0,xo,Xobs