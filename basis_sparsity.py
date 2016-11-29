import numpy as np
import spams
from scipy.spatial import distance
from scipy.stats import spearmanr, entropy
from union_of_transforms import double_sparse_nmf,smaf
import sys

THREADS = 5
MAX_BASIS = 1000
MIN_BASIS = 10
ERROR_THRESH = 0.005
MIN_FIT = 0.90

if __name__ == "__main__":
	inpath = sys.argv[1]
	xa = np.load(inpath)
	thresh = np.percentile(xa,99.5)
	xa[(xa > thresh)] = thresh
	u,s,vt = np.linalg.svd(xa,full_matrices=False)
	variance_fraction = np.cumsum(s**2/(s**2).sum())
	basis_size = np.where(variance_fraction > 1-ERROR_THRESH)[0][0]
	basis_size = min(MAX_BASIS,basis_size)
	ua = u[:,:basis_size]
	va = np.diag(s[:basis_size]).dot(vt[:basis_size])
	xh_svd = ua.dot(va)
	fit_svd = 1 - np.linalg.norm(xa-xh_svd)**2/np.linalg.norm(xa)**2
	module_size_svd = np.array([np.exp(entropy(abs(x))) for x in ua.T])
	usage_svd = np.array([np.exp(entropy(abs(x))) for x in va.T])
	ua_svd = ua
	w_svd = va
	ds = basis_size
	while True:
		ua,va = spams.nmf(np.asfortranarray(xa),return_lasso=True,K=ds,numThreads=THREADS)
		va = np.asarray(va.todense())
		x0 = ua.dot(va)
		d = 1 - np.linalg.norm(xa-x0)**2/np.linalg.norm(xa)**2
		if (d > 1-ERROR_THRESH) or (ds > xa.shape[1]*.4) or (ds > MAX_BASIS):
			break
		ds += max(1,ds/10)
	xh_nmf = ua.dot(va)
	fit_nmf = 1 - np.linalg.norm(xa-xh_nmf)**2/np.linalg.norm(xa)**2
	module_size_nmf = np.array([np.exp(entropy(abs(x))) for x in ua.T])
	usage_nmf = np.array([np.exp(entropy(abs(x))) for x in va.T])
	ua_nmf = ua
	w_nmf = va
	k = min(int(xa.shape[1]*1.5),ds*4)
	k = min(k,MAX_BASIS)
	UW = (np.random.random((xa.shape[0],k)),np.random.random((k,xa.shape[1])))
	lda2 = ERROR_THRESH
	while True:
		U,W = smaf(xa,k,10,lda2,maxItr=10,use_chol=True,activity_lower=4.,module_lower=400,UW=UW,donorm=True,mode=1,mink=5)
		nz = np.nonzero(U.sum(0))[0]
		U = U[:,nz]
		W = W[nz]
		xh_smaf = U.dot(W)
		fit_smaf = 1 - np.linalg.norm(xa-xh_smaf)**2/np.linalg.norm(xa)**2
		if (len(nz) > MIN_BASIS) and (fit_smaf > MIN_FIT):
			break
		elif lda2 < ERROR_THRESH/16:
			break
		else:
			lda2 /= 2.
	module_size_smaf = np.array([np.exp(entropy(abs(x))) for x in U.T])
	usage_smaf = np.array([np.exp(entropy(abs(x))) for x in W.T])
	# inpath, then fit, dict size, average module size, average module activity for each of SVD, NMF, SMAF
	print '%s, %f, %d, %f, %f, %f, %d, %f, %f, %f, %d, %f, %f' % (inpath,
			fit_svd,basis_size,np.average(module_size_svd),np.average(usage_svd),
			fit_nmf,ds,np.average(module_size_nmf),np.average(usage_nmf),
			fit_smaf,U.shape[1],np.average(module_size_smaf),np.average(usage_smaf))
	prefix = inpath[:inpath.rfind('/')]
	if True:
		np.save('%s/SVD.U.npy' % prefix,ua_svd)
		np.save('%s/SVD.W.npy' % prefix,w_svd)
		np.save('%s/SVD.module_size.npy' % prefix,module_size_svd)
		np.save('%s/SVD.module_activity.npy' % prefix,usage_svd)
		np.save('%s/NMF.U.npy' % prefix,ua_nmf)
		np.save('%s/NMF.W.npy' % prefix,w_nmf)
		np.save('%s/NMF.module_size.npy' % prefix,module_size_nmf)
		np.save('%s/NMF.module_activity.npy' % prefix,usage_nmf)
		np.save('%s/SMAF.U.npy' % prefix,U)
		np.save('%s/SMAF.W.npy' % prefix,W)
		np.save('%s/SMAF.module_size.npy' % prefix,module_size_smaf)
		np.save('%s/SMAF.module_activity.npy' % prefix,usage_smaf)
