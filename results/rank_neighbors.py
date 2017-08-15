import sys,pdb,os
import cPickle as pickle

PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
PATH_CACHE = os.path.join(BASE_DIR,'rank_neighbors_processor_cache.pkl')
SEED = 2052016
MAX_N = 1000

if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)
import results.process_results as process_results
import baseline.model_observables as model_observables

import pandas as pd
import numpy as np

class neighbor_ranking(object):
    def __init__(self,params=None,path_cache = PATH_CACHE,use_cache=False,dataset='KG_233'):

        #Load models/etc using the processor machinery, or from the cache
        if use_cache:
            self.processor = pickle.load(open(path_cache,'r'))
        else:
            self.processor = process_results.results(params)
            pickle.dump(self.processor,open(path_cache,'w'))

        #load data
        self.dataset = dataset
        self.load_dataset()
        self.n_entities = max(self.data.flatten())+1

        #load model_observable machinery as well
        self.model_observables = model_observables.model_observables()
        self.model_observables.load_dataset(dataset=self.dataset,VALIDATION_HOLDOUT = 0.05)
        self.model_observables.gen_training_data(N_TRAINING_SAMPLES=None,use_cache=True)
        self.model_observables.load_model()

        return

    def update_param(self,param):
        raise NotImplementedError

    def load_dataset(self):
        self.data = pickle.load(open(model_observables.paths_datasets[self.dataset],'r'))
        if self.dataset == 'KG_233':
            self.data,self.entity_to_idx = self.data
            self.idx_to_entity = {}
            for entity,idx in self.entity_to_idx.items():
                self.idx_to_entity[idx] = entity

        elif self.dataset == 'Wordnet':
            self.data,self.idx_to_entity,self.entity_to_idx = self.data

        return

    def idx_from_name(self,name):
        return self.entity_to_idx[name]

    def name_from_idx(self,idx):
        return self.idx_to_entity[idx]

    '''
    find list of closest neighbors
    idxs: iterable of node idxs
    labels: iterable of labels (already stored)
    returns: dict: labels --> array[idxs, scores for every other idx]
    '''
    def get_ranked_neighbors(self,idxs,labels):
        ranks = {}
        #retrieve scores for each type
        for label in labels:
            if self.processor.params[label]['reranking'] == 'Observable_rank':
                ranks[label] = self.compute_scores_observable(idxs)
            elif self.processor.params[label]['reranking'] == 'Embedding_rank':
                ranks[label] = self.compute_distances_embedding(idxs,label)
            elif self.processor.params[label]['reranking'].startswith('Observable_prerank'):
                ranks[label] = self.compute_distances_reranked(idxs,label)

            #sort
            ranks[label] = np.argsort(ranks[label],axis=1)

        return ranks

    '''
    computes vector of (negative) scores using observable model for each of the nodes in nodes.
    returns array of score vectors, indexed by nodes
    nodes can be strings or idxs (ints)
    '''
    def compute_scores_observable(self,idxs):
        scores = np.zeros((len(idxs),self.n_entities))
        for i,idx in enumerate(idxs):
            scores[i]=-self.model_observables.score_neighbors(idx)
        return scores

    '''
    computes vector of distances using the embeddings in labels (corresponding to params that have been loaded)
    returns array[idxs,distances_to_all_nodes]
    assume that labels correspond to params that are already loaded
    '''
    def compute_distances_embedding(self,idxs,label):
        param_dict = self.processor.params[label]
        model = self.processor.models[label]
        emb = model.binary_embeddings
        return get_distance_vectors(emb=emb,idxs=idxs,inversion=model.isInverted(),mapping=param_dict['mapping'])

    def compute_distances_reranked(self,idxs,label):
        #first, compute observable scores
        observable_scores = self.compute_scores_observable(idxs)

        #next, compute embedding distances
        embed_distances = self.compute_distances_embedding(idxs,label)

        #preranking
        reranked_distances = embed_distances
        #first, get relevant parameters out of processor
        param_dict = self.processor.params[label]
        prerank_thresh= int(param_dict['reranking'].rsplit(':')[1])
        mapping = param_dict['mapping']
        dim = param_dict['quantized_dim']

        #define filter
        prerank_filter = embed_distances < float(prerank_thresh)/dim
        sorted_idx = embed_distances.argsort(axis=1)
        for i,idx in enumerate(idxs):
            prerank_filter[i,sorted_idx[i,MAX_N:]] = False

        #apply filter
        if np.any(prerank_filter):
            reranked_distances[prerank_filter] = observable_scores[prerank_filter]

        return reranked_distances

def get_distance_vectors(emb,idxs,inversion,mapping='Hamming'):
    distances = np.zeros((len(idxs),emb.shape[0]))
    if mapping == 'L2':
        metric = lambda vec,mat: np.sqrt(np.mean((vec-mat)**2,axis=1))
    else:
        metric = lambda vec,mat: np.mean(np.abs(vec-mat),axis=1)

    for i,idx in enumerate(idxs):
        distances[i,:] = metric(emb[idx],emb)

    if inversion:
        return 1 - distances
    else:
        return distances






'''
Testing!
'''
if __name__=='__main__':

    #setup parameters
    params = {}
    dataset = 'KG_233'
    params[dataset]={}
    exp = {}
    fiedlerargs =['random_walk',100]
    quant = {}

    exp['KG_233']='S36'
    exp['Wordnet']='DG164'
    exp['Slashdot']='DG30b'
    if exp[dataset]:
        params[dataset]['Bernoulli Observable'] = {
                      'dataset':dataset,
                      'reranking':'Observable_rank',
                      'exp':exp[dataset],
                      'quantization':'Sample',
                      'quantized_dim':25,
                      'apply_sigmoid':True,
                      'mapping':'Hamming',
                      'loss':'MAP',
                      'loss_set':'test',
                      'P/R points':None,
                      'seed':2052016,
                      'n_precision_samples':1
        }
        params[dataset]['Bernoulli Sampled'] = {
                      'dataset':dataset,
                      'reranking':'Embedding_rank',
                      'exp':exp[dataset],
                      'quantization':'Sample',
                      'quantized_dim':25,
                      'apply_sigmoid':True,
                      'mapping':'Hamming',
                      'loss':'MAP',
                      'loss_set':'test',
                      'P/R points':None,
                      'seed':2052016,
                      'n_precision_samples':1
        }
        params[dataset]['Bernoulli Reranked'] = {
                      'dataset':dataset,
                      'reranking':'Observable_prerank_ball:3',
                      'exp':exp[dataset],
                      'quantization':'Sample',
                      'quantized_dim':25,
                      'apply_sigmoid':True,
                      'mapping':'Hamming',
                      'loss':'MAP',
                      'loss_set':'test',
                      'P/R points':None,
                      'seed':2052016,
                      'n_precision_samples':1
        }


    exp['KG_233']='DGb10'
    exp['Wordnet']='DG23'
    quant['KG_233']='SH'
    quant['Wordnet']='SH'
    if exp[dataset]:    
        params[dataset]['L2emb quantized'] = {
                      'dataset':dataset,
                      'reranking':'Embedding_rank',
                      'exp':exp[dataset],
                      'quantization':quant[dataset],
                      'quantized_dim':25,
                      'apply_sigmoid':True,
                      'mapping':'Hamming',
                      'loss':'MAP',
                      'loss_set':'test',
                      'P/R points':None,
                      'seed':2052016,
                      'n_precision_samples':1
        }

    # exp['KG_233']='DW14'
    # exp['Wordnet']='DW21'
    # quant['KG_233']='RHP'
    # quant['Wordnet']='SH'
    # params[dataset]['Deepwalk quantized'] = {
    #               'dataset':dataset,
    #               'reranking':'Embedding_rank',
    #               'exp':exp[dataset],
    #               'quantization':quant[dataset],
    #               'quantized_dim':25,
    #               'apply_sigmoid':False,
    #               'mapping':'Hamming',
    #               'loss':'MAP',
    #               'loss_set':'test',
    #               'P/R points':None,
    #               'seed':2052016,
    #               'n_precision_samples':1
    # }

    # quant['KG_233']='RHP'
    # quant['Wordnet']='RHP'
    # params[dataset]['Fiedler quantized'] = {
    #               'fiedler':fiedlerargs,
    #               'dataset':dataset,
    #               'reranking':'Embedding_rank',
    #               'quantization':quant[dataset],
    #               'quantized_dim':25,
    #               'apply_sigmoid':False,
    #               'mapping':'Hamming',
    #               'loss':'MAP',
    #               'loss_set':'test',
    #               'P/R points':None,
    #               'seed':2052016,
    #               'n_precision_samples':1
    # }

    # exp['KG_233']='DGb10'
    # exp['Wordnet']='DG23'
    # exp['Slashdot']='DG34'
    # quant['KG_233'] ='RHP'
    # quant['Wordnet']='ITQ'
    # if exp[dataset]:
    #     params[dataset]['L2emb quantized reranked'] = {
    #                   'dataset':dataset,
    #                   'reranking':'Observable_prerank_ball:3',
    #                   'exp':exp[dataset],
    #                   'quantization':quant[dataset],
    #                   'quantized_dim':25,
    #                   'apply_sigmoid':True,
    #                   'mapping':'Hamming',
    #                   'loss':'MAP',
    #                   'loss_set':'test',
    #                   'P/R points':None,
    #                   'seed':2052016,
    #               'n_precision_samples':1
    #     }

    # exp['KG_233']='DW14'
    # exp['Wordnet']='DW21'
    # exp['Slashdot']='DW7'
    # quant['KG_233']='RHP'
    # quant['Wordnet']='RHP'
    # params[dataset]['Deepwalk quantized reranked'] = {
    #               'dataset':dataset,
    #               'reranking':'Observable_prerank_ball:3',
    #               'exp':exp[dataset],
    #               'quantization':quant[dataset],
    #               'quantized_dim':25,
    #               'apply_sigmoid':False,
    #               'mapping':'Hamming',
    #               'loss':'MAP',
    #               'loss_set':'test',
    #               'P/R points':None,
    #               'seed':2052016,
    #               'n_precision_samples':1
    # }
    
    # quant['KG_233']='RHP'
    # quant['Wordnet']='RHP'
    # params[dataset]['Fiedler quantized reranked'] = {
    #               'fiedler':fiedlerargs,
    #               'dataset':dataset,
    #               'reranking':'Observable_prerank_ball:3',
    #               'quantization':quant[dataset],
    #               'quantized_dim':25,
    #               'apply_sigmoid':False,
    #               'mapping':'Hamming',
    #               'loss':'MAP',
    #               'loss_set':'test',
    #               'P/R points':None,
    #               'seed':2052016,
    #               'n_precision_samples':1
    # }

    ranker = neighbor_ranking(params[dataset],use_cache=True)
    
    test_entities = ['BARACK OBAMA_PERSON',
                     'MICROSOFT_ORGANIZATION',
                     'STEVE JOBS_PERSON',
                     'NEW DELHI_GPE',
                     'BRUCE WAYNE_PERSON',
                     'IBM_ORGANIZATION',
                     'ROGER FEDERER_PERSON',
                     'THE LORD OF THE RINGS_TITLEWORK']
    
    test_idxs = [int(ranker.idx_from_name(name)) for name in test_entities]
    test_rank = ranker.get_ranked_neighbors(idxs=test_idxs,labels=params[dataset].keys())    

    degree_1_connections = ranker.processor.graphs[dataset]['train'][np.array(test_idxs)]
    labels_of_interest = ['Bernoulli Observable','Bernoulli Reranked','Bernoulli Sampled','L2emb quantized']
    def print_top_no_connection(labels):
        for i in range(len(test_idxs)):
            print ' '
            print ranker.name_from_idx(test_idxs[i])
            entity = test_entities[i]
            for label in labels:
                print ' '
                print label
                j = -1
                k = 0
                while k < 10:
                    j+=1
                    jdx = test_rank[label][i,j]
                    if degree_1_connections[i,jdx] == 0:
                        k+=1
                        print j,ranker.name_from_idx(jdx)



    def print_top(test_ranks,test_entities):
        for idx,entity in enumerate(test_entities):
            print ' '
            print entity
            for label in test_ranks:
                print ' '
                print label
                for rank in range(10):
                    print ranker.name_from_idx(test_ranks[label][idx,rank])



    # test_observables = ranker.compute_scores_observable([9195])
    # test_embeddings = ranker.compute_distances_embedding([9195],'Bernoulli Sampled')
    # test_rerank = ranker.compute_distances_reranked([9195],'Bernoulli Reranked')
    # test_rank = ranker.get_ranked_neighbors(idxs=[9195],labels=['Bernoulli Observable','Bernoulli Reranked','Bernoulli Sampled'])

    # for rank in range(10): 
    #     print ranker.name_from_idx(test_rank['Bernoulli Observable'][0][rank])

    # for rank in range(10): 
    #     print ranker.name_from_idx(test_rank['Bernoulli Sampled'][0][rank])

    # for rank in range(10):
    #     print ranker.name_from_idx(test_rank['Bernoulli Reranked'][0][-1-rank])

    # for label in test_rank:
    #     print label
    #     for rank in range(10):
    #         print ranker.name_from_idx(test_rank[label][0][rank])