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

if __name__ == "__main__":
	data_path,measurements,sparsity,dictionary_size,training_dictionary_fraction = sys.argv[1:]
	print 'measurements: %s, sparsity: %s, dictionary size: %s, training fraction: %s' % (measurements,
			sparsity,dictionary_size,training_dictionary_fraction)
	measurements = int(measurements)
	sparsity = int(sparsity)
	dictionary_size = int(dictionary_size)
	X0 = np.load(data_path)
	# report usage of canonical basis vectors
	try:
		SVD_usage = np.load(data_path.replace('data.npy','diversity.SVD.npy'))
	except:
		u0,s0,vt0 = np.linalg.svd(X0)
		z = np.diag(s0).dot(vt0[:len(s0)])
		SVD_usage = np.array([np.exp(entropy(abs(x))) for x in z.T])
		np.save(data_path.replace('data.npy','diversity.SVD.npy'),SVD_usage)
	print '25th pctile SVD usage: %.2f, 50th pctile: %.2f, 75th pctile: %.2f' % (np.percentile(SVD_usage,25),
			np.percentile(SVD_usage,50),np.percentile(SVD_usage,75))
	# train bases
	training_dictionary_size = int(float(training_dictionary_fraction)*X0.shape[1])
	xi = np.zeros(X0.shape[1],dtype=np.bool)
	xi[np.random.choice(range(len(xi)),training_dictionary_size,replace=False)] = True
	xa = X0[:,xi]
	xb = X0[:,np.invert(xi)]
	ua,sa,vta = np.linalg.svd(xa,full_matrices=False)
	ua = ua[:,:min(dictionary_size,xa.shape[1])]
	X1,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua)
	print 'Performance in training samples with SVD:'
	spectral_compare(xa,X1,pc_n=0)
	X1,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua)
	print 'Performance in testing samples with SVD:'
	spectral_compare(xb,X1,pc_n=0)
	ua,va = spams.nmf(np.asfortranarray(xa),return_lasso=True,K=dictionary_size,clean=True,numThreads=THREADS)
	X2,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua)
	print 'Performance in training samples with NMF:'
	spectral_compare(xa,X2,pc_n=0)
	X2,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua)
	print 'Performance in testing samples with NMF:'
	spectral_compare(xb,X2,pc_n=0)
	# testing performance for broadly / narrowly expressed genes
	sample_diversity = np.array([np.exp(entropy(x)) for x in xb])/xb.shape[1]
	broad = (sample_diversity > 0.8)
	narrow = (sample_diversity > 0.001)*(sample_diversity < 0.2)
	print 'Testing performance on broadly-expressed genes ( > 80pct of samples): %.4f' % (1 - distance.correlation(xb[broad].flatten(),X2[broad].flatten()))
	print 'Testing performance on narrowly-expressed genes ( < 20pct of samples, > 0pct): %.4f' % (1 - distance.correlation(xb[narrow].flatten(),X2[narrow].flatten()))
