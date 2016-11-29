import numpy as np
from scipy.stats import entropy
import os,sys

#export PATH=$PATH:~/src/homer/bin/
module_path,genes_path,outpath,idx = sys.argv[1:]
idx = int(idx)

U = np.load(module_path)
#module_size_smaf = np.exp(entropy(abs(U[:,idx])))
xs = np.argsort(U[:,idx]**2)[::-1]
z = np.cumsum(U[xs,idx]**2)
module_size_smaf = np.where(z > 0.5)[0][0]
Gl = np.load(genes_path)
if not os.path.isfile('%s/all_genes.txt' % outpath):
	all_genes = Gl[(abs(U).sum(1) > 0)]
	f = open('%s/all_genes.txt' % outpath,'w')
	for g in all_genes:
		f.write(g.upper() + '\n')
	f.close()

genes = Gl[xs[:int(module_size_smaf)]]
os.system('mkdir %s/%s' % (outpath,idx))
f = open('%s/%s/genes.txt' % (outpath,idx),'w')
for g in genes:
	f.write(g.upper() + '\n')
f.close()
os.system('findGO.pl %s/%s/genes.txt human %s/%s/ -cpu 4 -bg %s/all_genes.txt' % (outpath,idx,outpath,idx,outpath))
