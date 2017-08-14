import numpy as np
from union_of_transforms import random_submatrix
from dl_simulation import *
from analyze_predictions import *
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sparse_optimization import SparseOptimization

def get_variable_Phi_y(Phi_f,Y_f,pv,g,s,X0,snr):
	pf = Phi_f.shape[0]
	m = pf + pv
	Phi = np.zeros((s,m,g))
	Y = np.zeros((m,s))
	for i in range(s):
		phi,y = rand_phi_y((pv,g,X0[:,i:i+1],snr))
		Phi[i,pf:] = phi
		Phi[i,:pf] = Phi_f
		Y[pf:,i] = y
	Y[:pf] = Y_f
	return Phi,Y

def rand_phi_y(args):
	m,g,x0,snr = args
	np.random.seed()
	phi = random_phi(m,g)
	y = get_observations(x0,phi,snr=snr)[:,0]
	return phi,y

def get_W(Y_variable,Phi_variable,D,n,k=5):
	W = np.array([sparse_decode(Y_variable[:,i:i+1],Phi_variable[i].dot(D),k,mink=min(5,k-1))[:,0] for i in range(n)]).T
	return W

def summarize_results(log_path,outpath):
	Results = defaultdict(list)
	result_labels = ['pearson_overall','spearman_overall','gene_pearson_avg','sample_pearson_avg','var_explained']
	FP = glob.glob(os.path.join(log_path,'BCS.o*'))
	for fp in FP:
		f = open(fp)
		for line in f:
			if 'data' in line:
				ls = line.strip().split()
				dataset = ls[0]
				dataset = dataset[:dataset.rfind('/')].replace('/','-')
				g,s,o,m = ls[1:5]
				pear,spear,g_pear,s_pear,ve = [float(x) for x in ls[5:10]]
				key = (dataset,g,s,o,m)
				Results[key].append((pear,spear,g_pear,s_pear,ve))
		f.close()
	Result_by_params = defaultdict(dict)
	Result_by_params_std = defaultdict(dict)
	for key,values in Results.items():
		dataset,g,s,o,m = key
		value_avg = np.average(values,axis=0)
		value_std = np.std(values,axis=0)
		Result_by_params[(g,o,m)][dataset] = value_avg
		Result_by_params_std[(g,o,m)][dataset] = value_std
	for params,datasets in Result_by_params.items():
		g,o,m = params
		f = open('%s/average_values.%s_g.%s_o.%s_m.txt' % (outpath,g,o,m),'w')
		f.write('\t'.join(['Dataset'] + result_labels) + '\n')
		for dataset,value_avgs in datasets.items():
			if 'Macosko' not in dataset:
				f.write('\t'.join([dataset]+[str(x) for x in value_avgs]) + '\n')
		f.close()
	for params,datasets in Result_by_params_std.items():
		g,o,m = params
		f = open('%s/stdev_values.%s_g.%s_o.%s_m.txt' % (outpath,g,o,m),'w')
		f.write('\t'.join(['Dataset'] + result_labels) + '\n')
		for dataset,value_avgs in datasets.items():
			if 'Macosko' not in dataset:
				f.write('\t'.join([dataset]+[str(x) for x in value_avgs]) + '\n')
		f.close()

def get_cluster_modules(Phi,Y,d,pf,maxItr=5,lda=0.1):
	D = np.random.random((Phi.shape[2],d))
	D = D/np.linalg.norm(D,axis=0)
	W = get_W(Y,Phi,D,Y.shape[1],k=min(d,20))
	for itr in range(maxItr):
		D = cDL(Y,Phi,W,D,lda,pf,sample_average_loss=False)
		W = get_W(Y,Phi,D,Y.shape[1],k=min(d,20))
	return D,W

def cDL(Y,Phi_variable,W,U,lda1,pf,maxItr=40,with_prints=False,nonneg=True,forceNorm=True,sample_average_loss=False):
	snl = SparseOptimization()
	snl.Y = Y.flatten()[:,np.newaxis]
	snl.Ynorm = np.linalg.norm(Y)
	snl.U = U
	def get_yhat(U):
		uw = U.reshape(snl.U.shape).dot(W)
		yhat = np.zeros(Y.shape)
		for i in range(yhat.shape[1]):
			yhat[:,i] = Phi_variable[i].dot(uw[:,i])
		return yhat.flatten()[:,np.newaxis]
	def proximal_optimum(U,delta,nonneg=False,forceNorm=False):
		Z = U.reshape(snl.U.shape)
		if delta > 0:
			z = (Z - delta*np.sign(Z))*(abs(Z) > delta)
		else:
			z = Z
		if nonneg:
			z[(z < 0)] = 0
		elif hasattr(snl,'prox_bounds'):
			z = np.maximum(z,self.prox_bounds[0])
			z = np.minimum(z,self.prox_bounds[1])
		if forceNorm:
			z = z/np.linalg.norm(z,axis=0)
			z[np.isnan(z)] = 0
		return z.flatten()[:,np.newaxis]
	if sample_average_loss:
		def grad_U(U,resid):
			r = resid.reshape(Y.shape)
			wgrad = np.zeros(U.shape)
			for i in range(r.shape[1]):
				wgrad += np.outer(Phi_variable[i].T.dot(r[:,i]),W[:,i]).flatten()[:,np.newaxis]
			return wgrad
		def get_resid(Yhat):
			resid = (Yhat.reshape(Y.shape) - Y)
			resid_0 = (resid[pf:]**2).sum(0)**.5 + 1e-3
			resid[pf:] = resid[pf:]/resid_0/Y.shape[1]
			resid_1 = (resid[:pf]**2).sum(1)**.5 + 1e-3
			resid[:pf] = (resid[:pf].T/resid_1/Y.shape[0]).T
			return resid.flatten()[:,np.newaxis]*snl.Ynorm
		def simple_loss(U,lda1):
			Yhat = get_yhat(U).reshape(Y.shape)
			loss = np.average(((Yhat[pf:] - Y[pf:])**2).sum(0)**.5)
			loss += np.average(((Yhat[:pf] - Y[:pf])**2).sum(1)**.5)
			return loss*snl.Ynorm + lda1*abs(U).sum()
	else:
		def grad_U(U,resid):
			r = resid.reshape(Y.shape)
			wgrad = np.zeros(U.shape)
			for i in range(r.shape[1]):
				wgrad += np.outer(Phi_variable[i].T.dot(r[:,i]),W[:,i]).flatten()[:,np.newaxis]
			return wgrad
		def get_resid(Yhat):
			return Yhat - snl.Y
		def simple_loss(U,lda1):
			Yhat = get_yhat(U)
			loss = 0.5*np.linalg.norm(Yhat - snl.Y)**2
			return loss + lda1*abs(U).sum()
	snl.get_Yhat = get_yhat
	snl.get_grad = grad_U
	snl.get_resid = get_resid
	snl.simple_loss = simple_loss
	snl.proximal_optimum = proximal_optimum
	lda = lda1*np.linalg.norm(grad_U(U.flatten()[:,np.newaxis],snl.Y).reshape(U.shape))/np.product(U.shape)*(np.log(U.shape[1])/Y.shape[1])**.5
	U1 = snl.nonlinear_proxGrad(lda,U.flatten()[:,np.newaxis],maxItr=maxItr,with_prints=with_prints,fa_update_freq=1e6,nonneg=nonneg,forceNorm=forceNorm)
	snl = None
	return U1.reshape(U.shape)

if __name__ == "__main__":
	inpath,gsom = sys.argv[1:]
	g,s,o,m = [int(s) for s in gsom.split(',')]
	Z = np.load(inpath)
	X0,xo,Xobs = random_submatrix(Z,g,s,o)
	SNR=2
	pf = m/5
	Phi_fixed = random_phi(pf,X0.shape[0])
	Y_fixed = get_observations(X0,Phi_fixed,snr=SNR)
	# begin by clustering samples based on a fixed set of composite measurements
	A = (Y_fixed/np.linalg.norm(Y_fixed,axis=0)).T
	dA = 1 - A.dot(A.T)
	dA = np.exp(-dA**2/2.)
	del A
	n = max(5,min(20,X0.shape[1]/50))
	lA = SpectralClustering(n_clusters=n,affinity='precomputed').fit_predict(dA)
	pv = m - pf
	Phi_variable,Y_variable = get_variable_Phi_y(Phi_fixed,Y_fixed,pv,X0.shape[0],X0.shape[1],X0,SNR)
	U = np.zeros((X0.shape[0],0))
	W = np.zeros((0,X0.shape[1]))
	dict_lda = 50.0
	# for full data (g=14202):
	#dict_lda = 5000.
	for c in set(lA):
		cidx = np.where(lA == c)[0]
		if X0.shape[1] > 1000:
			d = max(5,len(cidx)/20)
		else:
			d = max(5,len(cidx)/10)
		phi = Phi_variable[cidx]
		y = Y_variable[:,cidx]
		u,wc = get_cluster_modules(phi,y,d,pf,lda=dict_lda)
		del phi,y
		U = np.hstack([U,u])
		w = np.zeros((wc.shape[0],X0.shape[1]))
		w[:,cidx] = wc
		W = np.vstack([W,w])
		#x1 = u.dot(wc)
		#pearson,spearman,gene_pearson,sample_pearson,pc_dist = correlations(X0[:,cidx],x1,0)
		#var_fit = 1-np.linalg.norm(X0[:,cidx]-x1)**2/np.linalg.norm(X0[:,cidx])**2
		#uent = np.average([np.exp(entropy(u)) for u in U.T])
		#print inpath,c,X0.shape[0],len(cidx),o,m,pearson,spearman,gene_pearson,sample_pearson,var_fit,uent
	X2 = U.dot(W)
	X2[(X2 < 0)] = 0
	pearson,spearman,gene_pearson,sample_pearson,pc_dist = correlations(X0,X2,0)
	var_fit = 1-np.linalg.norm(X0-X2)**2/np.linalg.norm(X0)**2
	print inpath,X0.shape[0],X0.shape[1],o,m,pearson,spearman,gene_pearson,sample_pearson,var_fit
	for _ in range(5):
		U = cDL(Y_variable,Phi_variable,W,U,dict_lda,pf,sample_average_loss=False)
		W = get_W(Y_variable,Phi_variable,U,X0.shape[1],k=20)
		X2 = U.dot(W)
		X2[(X2 < 0)] = 0
		pearson,spearman,gene_pearson,sample_pearson,pc_dist = correlations(X0,X2,0)
		var_fit = 1-np.linalg.norm(X0-X2)**2/np.linalg.norm(X0)**2
		uent = np.average([np.exp(entropy(u)) for u in U.T])
		print inpath,X0.shape[0],X0.shape[1],o,m,pearson,spearman,gene_pearson,sample_pearson,var_fit,uent
	# for full data:
	#U1,V1 = smaf(X2,U.shape[1],20,0.005,maxItr=5,use_chol=True,activity_lower=0.,module_lower=500,UW=(U,W),donorm=True,mode=1,mink=5.,doprint=True)
