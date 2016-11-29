import numpy as np
from union_of_transforms import smaf,random_submatrix
from dl_simulation import *
from analyze_predictions import *
from scipy.spatial import distance
from scipy.sparse.linalg import svds
import psutil
from multiprocessing import Pool
from time import time

def get_variable_Phi_y_x(m,g,s,X0,snr):
	pool = Pool(THREADS)
	Phi_v = np.zeros((s,m,g))
	Y_v = np.zeros((m,s))
	X1 = np.zeros(X0.shape)
	results = pool.map(rand_phi_yx,[(i,m,g,X0[:,i:i+1],snr) for i in range(s)], chunksize=(max(1,s/THREADS)))
	for i,phi,y,x in results:
		Phi_v[i] = phi
		Y_v[:,i] = y
		X1[:,i] = x
	pool.close()
	pool.join()
	return Phi_v,Y_v,X1

def rand_phi_yx(args):
	i,m,g,x0,snr = args
	np.random.seed()
	phi = random_phi(m,g)
	y = get_observations(x0,phi,snr=snr)[:,0]
	x = phi.T.dot(np.linalg.inv(phi.dot(phi.T))).dot(y)
	return i,phi,y,x

def get_clusters(X,k,ldafit1=0.06,ldafit2=1e-4):
	U,W = smaf(X,k,ldafit1,ldafit2,maxItr=10,module_lower=X.shape[0]/10,donorm=True,mode=1)
	#U,W,ldafit1,ldafit2 = initialize_from_smaf(X,k,ldafit1,ldafit2,ldamin=1e-8)
	X_cluster_support = []
	for i in range(k):
		genes = np.nonzero(U[:,i])[0]
		samples = np.nonzero(W[i])[0]
		X_cluster_support.append((genes,samples))
	return X_cluster_support

def get_dict_atoms(Y_variable,Phi_variable,X_cluster_support):
	k = len(X_cluster_support)
	X = []
	for i in range(Y_variable.shape[1]):
		if isinstance(Phi_variable,str):
			phi = np.load('%s/phi.%d.npy' % (Phi_variable,i))
		else:
			phi = Phi_variable[i]
		x = phi.T.dot(np.linalg.inv(phi.dot(phi.T))).dot(Y_variable[:,i])
		X.append(x)
	X = np.array(X).T
	U = []
	for i in range(k):
		genes,samples = X_cluster_support[i]
		if (len(genes) > 1) and (len(samples) > 1):
			x_supp = X[genes][:,samples]
			u_supp,s,vt = svds(x_supp,k=1)
			#u_supp,vt = spams.nmf(np.asfortranarray(x_supp),return_lasso=True,K = 1,numThreads=THREADS)
			u = np.zeros(X.shape[0])
			u[genes] = u_supp[:,0]
			U.append(u)
	U = np.array(U).T
	return U,X

def run_DL(X,U,lda1):
	Xnorm = np.linalg.norm(X)**2/X.shape[1]
	param = {'D': np.asfortranarray(U),'mode':1,'lambda1': lda1*Xnorm,'iter':20,'numThreads': THREADS}
	D = spams.trainDL_Memory(np.asfortranarray(X),**param)
	return D

def get_W(Y_variable,Phi_variable,D,n,k=5):
	if isinstance(Phi_variable,str):
		W = []
		for i in range(n):
			cache_info = psutil.phymem_usage()
			while cache_info.cached/float(cache_info.total) > 0.90:
				print 'sleeping to let cache clear?',cache_info.cached/float(cache_info.total)
				time.sleep(120)
				cache_info = psutil.phymem_usage()
			phi = np.load('%s/phi.%d.npy' % (Phi_variable,i))
			w = sparse_decode(Y_variable[:,i:i+1],phi.dot(D),k)[:,0]
			W.append(w)
		W = np.array(W).T
	else:
		W = np.array([sparse_decode(Y_variable[:,i:i+1],Phi_variable[i].dot(D),k)[:,0] for i in range(n)]).T
	return W

if __name__ == "__main__":
	inpath,gsom = sys.argv[1:]
	g,s,o,m = [int(s) for s in gsom.split(',')]
	Z = np.load(inpath)
	X0,xo,Xobs = random_submatrix(Z,g,s,o)
	SNR=2
	#pf = m/5
	#Phi_fixed = random_phi(pf,X0.shape[0])
	pf = 0
	pv = m - pf
	Phi_variable,Y_variable,X1 = get_variable_Phi_y_x(pv,X0.shape[0],X0.shape[1],X0,SNR)
	#Y_fixed = get_observations(X0,Phi_fixed,snr=SNR)
	d=50
	#X1 = Phi_fixed.T.dot(np.linalg.inv(Phi_fixed.dot(Phi_fixed.T))).dot(Y_fixed)
	X1[xo] = Xobs
	# for very large datasets generate in parallel with random_phi.py
	if False:
		Y_variable = np.array([np.load('TMP_PATH/yv.%d.npy' % i) for i in range(X0.shape[1])]).T
		X1 = np.array([np.load('TMP_PATH/x1.%d.npy' % i) for i in range(X0.shape[1])]).T
		Phi_variable = 'TMP_PATH'
		d = 500
	for itr in range(2):
		# for a small number of genes:
		#U,W = smaf(X1,d,15,0.005,maxItr=10,activity_lower=0.,module_lower=min(400,X1.shape[0]/20),donorm=True,mode=1,use_chol=True,mink=5,doprint=True)
		# for a larger number of genes:
		#U,W = smaf(X1,d,20,0.5,maxItr=5,activity_lower=0.,module_lower=min(200,X1.shape[0]/20),donorm=True,mode=1,use_chol=True,mink=15,doprint=True)
		UW = (np.random.random((X1.shape[0],d)),np.random.random((d,X1.shape[1])))
		U,W = smaf(X1,d,20,0.1,maxItr=1,use_chol=True,activity_lower=0.,module_lower=min(200,X1.shape[0]/20),UW=UW,donorm=True,mode=1,mink=10,doprint=True)
		print (U.sum(0) > 0).sum(),(abs(W).sum(1) > 0).sum()
		W0 = np.copy(W)
		D = run_DL(X1,U,0.05)
		W = get_W(Y_variable,Phi_variable,D,X0.shape[1],k=15)
		X1 = D.dot(W)
		#X1[xo] = Xobs
		X1[(X1 < 0)] = 0
		pearson,spearman,gene_pearson,sample_pearson,pc_dist = correlations(X0,X1,0)
		var_fit = 1-np.linalg.norm(X0-X1)**2/np.linalg.norm(X0)**2
		print inpath,X0.shape[0],X0.shape[1],o,m,pearson,spearman,gene_pearson,sample_pearson,var_fit
		if itr == 0:
			u0,w0,d0,w0b = np.copy(U),W0,np.copy(D),np.copy(W)
	UW = (np.random.random((X1.shape[0],d)),np.random.random((d,X1.shape[1])))
	U1,W1 = smaf(X1,d,20,0.1,maxItr=1,use_chol=True,activity_lower=0.,module_lower=min(200,X1.shape[0]/20),UW=UW,donorm=True,mode=1,mink=10,doprint=True)
