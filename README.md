# CS-SMAF

This code accompanies the following publication and was used to generate results therein:

B Cleary, L Cong, E Lander, A Regev. "Composite measurements and molecular compressed sensing for highly efficient transcriptomics" (2016)

While this code is not packaged as a proper pipeline, it contains several routines that we feel will be useful in broader research contexts. These are listed below.

## Sparse Module Activity Factorization (SMAF)
Within our scripts SMAF is defined in union_of_transforms.py. However, since SMAF will likely have utility independent of compressed sensing contexts, we have also generated a dedicated file (smaf.py), with more detailed instructions for computing the factorization.

## Compressed recovery of gene expression with a known dictionary
The general routine we used for simulating compressed sensing is as follows:
  1. Compute a dictionary (U) from training data: e.g. U,W = smaf(Xtraining)
  2. In testing data, simulate noisy compressed observations (Y)
  3. Use Y and the measurement matrix A to decode module activity levels: i.e. W = sparse_decode(Y,A.dot(U),k)
  4. Recover the high-dimensional expression levels: i.e. X_hat = U.dot(W)

This process is implemented in direct_vs_distributed.py using various matrix factorization techniques in step 1. Steps 2-4 are carried out by the function "recover_system_knownBasis", which can be found in dl_simulation.py

## Blind compressed sensing with SMAF (BCS-SMAF)
Our methods for blindly recovering gene expression from compressed observations without any training data can be found in BCS.py. Note that there are slightly different implementations depending on data size. These were need to efficiently (in terms of both memory and time) implement BCS-SMAF on very large datasets (e.g. GTEx).
