import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy

class SparseOptimization(object):
	def __init__(self):
	    self.totally_radical = True
	
	def nonlinear_proxGrad(self,lda1,Z,maxItr=100,tol=1e-5,step_factor=2,acceptance_const=0.5,crit_size=1,eps=1e-3,with_prints=False,fa_update_freq=10,nonneg=False,forceNorm=False,printFreq=10):
		Z_series = [np.copy(Z)]
		grad_series = []
		for itr in range(maxItr):
			Z = Z_series[-1]
			if hasattr(self,'Yesh'):
				grad = self.get_grad(Z)
			else:
				Yhat = self.get_Yhat(Z)
				resid = self.get_resid(Yhat)
				grad = self.get_grad(Z,resid)
			grad_series.append(grad)
			if forceNorm:
				step = self.get_step(Z_series,grad_series,min_step=1e-5)
			else:
				step = self.get_step(Z_series,grad_series)
			# we redo the last crit_size losses each itr because self.fa might have changed
			loss_series = []
			for j in range(max(0,itr-crit_size),itr+1):
			    loss_series.append(self.simple_loss(Z_series[j],lda1))
			while True:
				update = Z - grad/step
				z = self.proximal_optimum(update,lda1/step,nonneg=nonneg,forceNorm=forceNorm)
				# acceptance criterion
				cond = 0
				ac = acceptance_const*step/2*np.linalg.norm(z - Z)**2
				for l in loss_series:
				    c = l - ac
				    if c > cond:
				        cond = c
				#print itr,step,self.simple_loss(z,lda1),cond,loss_series,ac
				if (cond > 0) and (self.simple_loss(z,lda1) <= cond*(1+eps)):
				    Z_series.append(z)
				    break
				step *= step_factor
				if step > 2**100:
					break
			if step > 2**100:
				print 'FAILURE: prox grad step size -> 0. Resetting to initial value...'
				Z = Z_series[0]
				if (itr < fa_update_freq) and (fa_update_freq < maxItr):
					print 'updating fa'
					self.update_fa(Z)
				break
			Z = Z_series[-1]
			if (itr > 0) and (itr%fa_update_freq == 0):
			    print 'updating fa'
			    self.update_fa(Z)
			if with_prints and (itr%printFreq == 0):
			    self.print_performance(Z,itr,Z_series)
			if ((np.linalg.norm(Z_series[-2] - Z_series[-1])/np.linalg.norm(Z_series[-2]) < tol) or (np.linalg.norm(Z_series[-2])==0)) and (itr > 2):
				if (itr < fa_update_freq) and (fa_update_freq < maxItr):
					print 'updating fa'
					self.update_fa(Z)
				break
		return Z
	
	def get_resid(self,Yhat):
		Y = self.Y
		resid = ne.evaluate("Yhat - Y")
		if len(Yhat.shape) == 2:
			# assume we are shape: (s x h) x e
			resid_0 = ne.evaluate("resid**2").sum(0)**.5 + 1e-3
			resid = ne.evaluate("resid/resid_0")
		elif len(Yhat.shape) == 3:
			# assume we are shape: e x s x h
			resid_0 = np.linalg.norm(np.linalg.norm(resid,axis=1),axis=1) + 1e-3
			resid = (resid.T/resid_0).T
		return resid
	
	def print_performance(self,Z,itr,Z_series=(1.,1.)):
		Yhat = self.get_Yhat(Z)
		fit = 1 - np.linalg.norm(self.Y - Yhat)**2/np.linalg.norm(self.Y)**2
		r2 = (1 - distance.correlation(self.Y.flatten(),Yhat.flatten()))**2
		print 'itr: %d, fit: %f, r2: %f, Z entropy: %f, Z min: %f, Z max: %f, Z change: %f' % (itr,fit,r2,
		np.average([np.exp(entropy(abs(z))) for z in Z.T]),Z.min(),Z.max(),
		np.linalg.norm(Z_series[-2] - Z_series[-1])/np.linalg.norm(Z_series[-2]))
		self.fit = (fit,r2)
	
	def get_fit(self):
		W = self.get_W()
		Yhat = ttm(W,self.E,0)
		#Yhat = ttm(Yhat,self.S,1)
		Yhat = ttm(Yhat,self.H,2)
		fit = 1 - np.linalg.norm(self.Y - Yhat)**2/np.linalg.norm(self.Y)**2
		fitavg = np.average(np.linalg.norm(np.linalg.norm(self.Y - Yhat,axis=1),axis=1))
		r2 = (1 - distance.correlation(self.Y.flatten(),Yhat.flatten()))**2
		return fit,r2,fitavg
	
	def get_step(self,Z_series,grad_series,min_step=0.0001):
	    if len(grad_series) > 1:
	        d = Z_series[-1] - Z_series[-2]
	        g = grad_series[-1] - grad_series[-2]
	        #return min((d.T.dot(g)).sum()/(d.T.dot(d)).sum() , (g.T.dot(g)).sum()/(d.T.dot(g)).sum())
	        return max(min_step,min((d*g).sum()/(d*d).sum() , (g*g).sum()/(d*g).sum())/1000.)
	    else:
	        return min_step
	
	def simple_loss(self,Z,lda1):
		Yhat = self.get_Yhat(Z)
		if len(Yhat.shape) == 2:
			#loss = np.average(np.linalg.norm(resid,axis=0))
			Y = self.Y
			loss = np.average(ne.evaluate("(Yhat - Y)**2").sum(0)**.5)
		elif len(Yhat.shape) == 3:
			resid = Yhat - self.Y
			loss = np.average(np.linalg.norm(np.linalg.norm(resid,axis=1),axis=1))
		return loss + lda1*abs(Z).sum()
	
	def proximal_optimum(self,Z,delta,nonneg=False,forceNorm=False):
		if delta > 0:
			z = (Z - delta*np.sign(Z))*(abs(Z) > delta)
		else:
			z = Z
		if nonneg:
			z[(z < 0)] = 0
		elif hasattr(self,'prox_bounds'):
			z = np.maximum(z,self.prox_bounds[0])
			z = np.minimum(z,self.prox_bounds[1])
		if forceNorm:
			z = (z/np.linalg.norm(z,axis=0))
			z[np.isnan(z)] = 0
		return z