#Contains code both for generating binary / real embedding data (run by Vinith) and for performing nearest-neighbor retrieval usign the observables model (used by Sumit in observables ranking + reranking)

import baseline.model_observables as model_observables
import cPickle as pickle
import os
import numpy as np


DUMP_BASE_DIR = '/Users/vmisra/Desktop/sumit_data/'

# ##First, generate embeddings as CSV files for Sumit
# ## NOT for timing purposes --- just keeping
# embeddings = pickle.load(open('/Users/vmisra/kgdata/experiments/NCE_KG1/snapshots/params_best_sampled.pkl','r'))['params'][0]
# np.savetxt(open(os.path.join(DUMP_BASE_DIR, 'real_embeddings_10.csv.gz'),'w'), X=embeddings, delimiter='\t',newline='\n')
#
# embeddings_quant = (embeddings >= 0.5).astype(int)
# np.savetxt(open(os.path.join(DUMP_BASE_DIR, 'binary_embeddings_10.csv.gz'),'w'), X=embeddings_quant, delimiter='\t',newline='\n')
#
# #Next, for 25 bit embeddings...
# embeddings = pickle.load(open('/Users/vmisra/kgdata/experiments/NCE_KG0.5/snapshots/params_best_sampled.pkl','r'))['params'][0]
# np.savetxt(open(os.path.join(DUMP_BASE_DIR, 'real_embeddings_25.csv.gz'),'w'), X=embeddings, delimiter='\t',newline='\n')
#
# embeddings_quant = (embeddings >= 0.5).astype(int)
# np.savetxt(open(os.path.join(DUMP_BASE_DIR, 'binary_embeddings_25.csv.gz'),'w'), X=embeddings_quant, delimiter='\t',newline='\n')


##Observables based modeling
PATH_KG_DATA = '/Users/vmisra/kgdata/data/traindata_db233_minmentions10_minentity3.pkl'
N_TRAINING_SAMPLES = 100000
x = model_observables.model_observables()



#First, we load a model observables object.

print "loading data"
x.load_dataset(dataset=None,VALIDATION_HOLDOUT = 0.05, path_dataset=PATH_KG_DATA)

#Next, define functions to perform featurization for nearest neighbor retrieval (essentially, measures the speed of retrieval with the ranking model)
def observables_scoring_rank(idxs):
    #for each idx to perform NN retrieval for, featurize the entire graph of potential neighbors for retrieval purposes.
    features = model_observables.featurize_all(x.graphs['train'], idxs)
    return "SUCCESS"

##Next, featurization for reranking (giv
def observables_scoring_rerank(idxs, shortlist_list_of_arrays):
    #for each idx to perform reranking NN retrieval for, featurize the corresponding row of candidate NN's in shortlist matrix.
    features = model_observables.featurize_subset(x.graphs['train'],idxs,subset_list_of_arrays=shortlist_list_of_arrays)
    return "SUCCESS"

## testing
import time
print "timing ranking time for 2 samples"
start_time = time.time()
print observables_scoring_rank(idxs = np.array([10,20]))
print time.time()-start_time

print "timing reranking time for 2 samples and a subset of 10 indices"
start_time = time.time()
print observables_scoring_rank(idxs = np.array([10,20]), shortlist_list_of_arrays = [np.array([1,2]), np.array([3,4])])
print time.time()-start_time
