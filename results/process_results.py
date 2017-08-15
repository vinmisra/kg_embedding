PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
VAL_FRAC = 0.05
MAX_N = 1000

import sys,pdb,os
import cPickle as pickle
import copy
if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import theano
import theano.tensor as T

import hypers
from hypers import get_hypers
import models.model_testing
from models.model_testing import model_tester

from utils.est_entro import xlogx,est_entro_JVHW
from utils.utils import get_graph, get_graph_mean, get_graph_directed, get_graph_directed_mean
import utils.neg_sampling as neg_sampling

import models.model_fiedler as model_fiedler
import results.load_models_results
import results.load_models_results as load_models_results
reload(results.load_models_results)

class results(object):
    '''
    params: dict
            keys: names to be printed in association with each given experiment
            values: dict
                'dataset':
                    'KG_233', 'Wordnet', 'Slashdot' (use the given dataset)
                'reranking':
                    'Observable_prerank_top:N' (rerank the results of the embedding described below, after selecting the top N results for reranking)
                    'Observable_prerank_ball:D' (rerank the results of the embedding described below, after selecting the entries within radius D for reranking)
                    'Observable_rank' (only use observables to perform ranking)
                    'Embedding_rank' (only use the embeddings)
                'fiedler':
                    None: No fiedler embedding
                    [<laplacian_type>,<embedding_dim>] 
                        laplacian_type: 'unnormalized','symmetric','random_walk'
                        embedding_dim: dimensionality of the Fiedler embedding
                'exp': 
                    exp label
                'quantization': 
                    'Sample'
                    'Raw'
                    'SH'
                    'ITQ'
                    'RHP'
                'mapping': typed of mapping from embed vectors to probabilities
                    'Learned' (only for discrete)
                    'Hamming'
                    'L2'
                'apply_sigmoid':
                    True: apply sigmoid to raw embeddings before quantizing them
                'quantized_dim': How many dims to quantize to. None --> keep it the same
                'loss': 
                    <Name of entry from CSV, e.g. 'val loss sampled', 'val loss expected'>
                    'Log Loss' (compute)
                    'Peak F1' (compute)
                    'P/R' (compute) --- give spacing in additional parameter
                    'P/R AUC' (compute) --- give spacing in additional parameter
                'loss_set':
                    'train'
                    'test'
                    'all'
                'P/R points': vector of distances to sample P and R at.
                'seed': optional seed to use for quantizer + train/test split
                'n_precision_samples': number of samples to use to compute the precision
                'graph_symmetry': neighborhood relevant: directed or sym graph?
                'neighborhood_mean': nbhood relevant: normalized graph?
    '''
    def __init__(self, params, seed = 2052016):
        self.seed = seed #used for train/test split of datasets
        self.params = params
        self.recorded_params = load_models_results.load_results('/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano/results/results_data').dropna(axis=0,how='all')

        self.datasets = {}
        self.models = {}
        self.graphs = {}
        self.model_polarities = {}
        self.observable_test_ranks = {}
        self.observable_test_samples = {}
        self.observable_test_scores = {}
        #self.df = pd.DataFrame(columns=['label','log loss','f1','precision','recall','auc','loss set','val loss recorded'])

        self.new_results(params)

    def get_entries(self,label,param):
        new_entries = []
        print "generating results for ",label

        if ('exp' in param) and (param['exp'] in self.recorded_params.index):
            recorded_param = self.recorded_params.loc[param['exp']]
        else:
            recorded_param = None

        loss_type = param['loss']
        loss = self.get_loss(label,param,recorded_param) #loss is either a single value, or a list of precision/recall dictionaries.
        #if it's a list of dictionaries, e.g. [{'precision':_,'recall':_},{'precision:_,'recall':_},...], add an entry for each one
        if loss_type == 'P/R':
            for subloss in loss:
                if 'exp' in param:
                    subloss['exp'] = param['exp']
                subloss['label'] = label
                subloss['quant'] = param['quantization']
                subloss['dataset'] = param['dataset']
                new_entries.append(subloss)
        else:
            lossdict = {loss_type: loss, 'label': label,'quant': param['quantization'],'dataset': param['dataset']}
            if 'exp' in param:
                lossdict['exp'] = param['exp']
            new_entries.append(lossdict)

        return new_entries

    def new_results(self,params):
        new_entries = []

        #goes through each "experiment" to run and record results from
        for label,param in params.items():
            new_entries.extend(self.get_entries(label,param))

        self.df = pd.DataFrame(new_entries)
        if 'exp' in self.df.columns:
            self.df = pd.merge(self.df,self.recorded_params.reset_index(),left_on='exp',right_on='Name',how='left')

    def update_results(self,params):
        new_entries = []

        #first, delete pre-existing entries, in case this is an update
        for label,param in params.items():
            self.df = self.df[self.df['label']!=label]

        #then compute entries for the new params
        new_entries = []
        for label,param in params.items():
            new_entries.extend(self.get_entries(label,param))
        df_new = pd.DataFrame(new_entries)
        if 'exp' in df_new.columns:
            df_new = pd.merge(df_new,self.recorded_params.reset_index(),left_on='exp',right_on='Name',how='left')

        #finally, insert into self.df
        self.df = pd.concat([self.df,df_new],axis=0)
    '''
    depending on loss type, returns either a single loss or a list of dictionaries of loss (precision recall)
    '''
    def get_loss(self, label, param_dict,recorded_param):
        loss_type = param_dict['loss']
        reranking = param_dict['reranking']

        #load data and model
        if param_dict['dataset'] not in self.datasets:
            self.load_dataset(param_dict['dataset'])
            self.load_observable_rankings(param_dict['dataset'])
            self.load_observable_samples(param_dict['dataset'])
        if param_dict['reranking'].startswith('Observable_prerank'):
            if param_dict['dataset'] not in self.observable_test_scores:
                self.load_observable_scores(param_dict['dataset'])

        self.load_model(label,param_dict,recorded_param)

        ###and compute loss from "scratch"

        if loss_type == 'P/R':
            return self.get_loss_pr(label, param_dict, recorded_param, reranking)

        if loss_type == 'MAP':
            return self.get_loss_MAP(label,param_dict, recorded_param, reranking)

        if loss_type == 'P/R AUC':
            return self.get_loss_auc(label, param_dict)

        if loss_type == 'Peak F1':
            return self.get_loss_f1(label, param_dict)

        if loss_type == 'Log Loss':
            return self.get_loss_logloss(label, param_dict, recorded_param)

        if loss_type == 'Marginal Entropy':
            return self.get_loss_marginal_entropy(label)

        if loss_type == 'Joint Entropy Plugin':
            return self.get_loss_joint_entropy_plugin(label)

        else:
            return -1000 #code for no loss recognized.

    def get_loss_marginal_entropy(self, label):
        #first, get the embedding matrix
        emb = self.models[label].binary_embeddings
        #find probability of each bit...
        p_per_bit = emb.mean(axis=0)
        #and compute its entropy...
        entropy = np.mean(xlogx(p_per_bit)+xlogx(1-p_per_bit))
        return entropy

    def get_loss_joint_entropy_plugin(self, label):
        #first, get the embedding matrix
        emb = self.models[label].binary_embeddings
        #convert each to unique and small integer
        powers = 2**(np.arange(emb.shape[1]))
        emb_int = (emb*powers).sum(axis=1).astype(int)
        emb_int_uniques = np.unique(emb_int,return_inverse = True)[1]
        #compute distribution
        freqs = np.bincount(emb_int_uniques)
        dist = freqs/freqs.sum().astype(float)
        return np.sum(xlogx(dist))



    def get_loss_logloss(self, label, param_dict, recorded_param):
        model = self.models[label]
        data_test = self.datasets[param_dict['dataset']][param_dict['loss_set']]
        data_train = self.datasets[param_dict['dataset']]['train']
        if param_dict['mapping']=='Learned':
            model.train_logloss(idxs_train=data_train,lr=5,epochs=20,algorithm='ADAGRAD')
        return model.test_logloss(idxs_test=data_test)

    def get_loss_f1(self, label, param_dict):
        raise NotImplementedError


    '''
    returns a list of dictionaries, each with a "precision" and "recall" entry
    '''
    def get_loss_pr(self, label, param_dict, recorded_param,reranking):
        if reranking == 'Observable_rank':
            ''' simplest case: we've already computed the ranks and have them stored. '''
            precisions = get_PR_observable(self.observable_test_ranks[param_dict['dataset']],
                                           recalls = param_dict['P/R points'])
        else:
            model = self.models[label]
            data = self.datasets[param_dict['dataset']][param_dict['loss_set']]
            quantized_emb = model.binary_embeddings
            inverted = model.isInverted()

            if reranking.startswith('Observable_prerank'):
                precisions = get_PR_observable_prerank(observable_scores = self.observable_test_scores[param_dict['dataset']],
                                                       prerank_type=reranking,
                                                       samples = self.observable_test_samples[param_dict['dataset']],
                                                       emb=quantized_emb,
                                                       graphs=self.graphs[param_dict['dataset']],
                                                       recalls=param_dict['P/R points'],
                                                       inversion=inverted,
                                                       mapping=param_dict['mapping'])
            elif reranking == 'Embedding_rank':
                if 'n_precision_samples' not in param_dict:
                    param_dict['n_precision_samples'] = 100
                precisions= get_PR_testset(emb=quantized_emb,
                                           graphs=self.graphs[param_dict['dataset']],
                                           recalls=param_dict['P/R points'],
                                           inversion=inverted,
                                           mapping=param_dict['mapping'],
                                           n_samples=param_dict['n_precision_samples'])
            else:
                raise NotImplementedError
        
        pr_losses = []
        for precision,recall in zip(precisions,param_dict['P/R points']):
            pr_losses.append({'precision':precision,'recall':recall})

        return pr_losses

    def get_loss_MAP(self,label, param_dict, recorded_param,reranking):
        if 'n_precision_samples' not in param_dict:
                param_dict['n_precision_samples'] = 100

        if reranking == 'Observable_rank':
            return get_MAP_observable(self.observable_test_ranks[param_dict['dataset']])
        elif reranking == 'Unigram':
            return get_MAP_unigram(data_train=self.datasets[param_dict['dataset']]['train_sym'],
                                    graphs=self.graphs[param_dict['dataset']],
                                    n_samples=param_dict['n_precision_samples'])

        #otherwise, continue on...
        model = self.models[label]
        data = self.datasets[param_dict['dataset']][param_dict['loss_set']]
        quantized_emb = model.binary_embeddings
        inverted = model.isInverted()

        if reranking.startswith('Observable_prerank'):
            return get_MAP_observable_prerank(observable_scores=self.observable_test_scores[param_dict['dataset']],
                                              samples=self.observable_test_samples[param_dict['dataset']],
                                              prerank_type=reranking,
                                              emb=quantized_emb,
                                              graphs=self.graphs[param_dict['dataset']],
                                              inversion=inverted,
                                              mapping=param_dict['mapping'])
        else:
            return get_MAP_testset(quantized_emb,self.graphs[param_dict['dataset']],inversion=inverted,n_samples=param_dict['n_precision_samples'])

    def get_loss_auc(self, label, param_dict):
        raise NotImplementedError
    
    '''
    loads model via model_frozen_bitembeddings and quantization parameters in param_dict
    '''
    def load_model(self,label, param_dict,recorded_param):
        if param_dict['reranking'] == 'Observable_rank':
            return #no model to load in this case

        #load embeddings
        if 'fiedler' in param_dict and param_dict['fiedler'] != None:
            fiedler_model = model_fiedler.model_fiedler(graph=self.graphs[param_dict['dataset']]['train'],
                                                        dim=param_dict['fiedler'][1],
                                                        laplacian_type=param_dict['fiedler'][0])
            model_params = (fiedler_model.get_embeddings(),1,0)  #give fake values for original_a and original_b --- neither are used for deepwalk.
        elif type(recorded_param)!=type(None) and 'emb type' in recorded_param and recorded_param['emb type'] == 'deepwalk':
            model_params = load_models_results.load_deepwalk(recorded_param['dataset'],recorded_param['dim'])
            model_params = (model_params,1,0) #give fake values for original_a and original_b --- neither are used for deepwalk.
        else:
            if param_dict['quantization'] == 'Sample' or 'best_sampled' in param_dict:
                best_sampled = True
            else:
                best_sampled = False
            model_params = load_models_results.copy_and_load(param_dict['exp'],best_sampled)


        if 'seed' not in param_dict:
            param_dict['seed'] = None    
        
        if 'graph_symmetry' not in param_dict or param_dict['graph_symmetry']==False:
            if 'neighborhood_mean' in param_dict and param_dict['neighborhood_mean']==True:
                graph_train = self.graphs[param_dict['dataset']]['train_directed_mean']
            else:
                graph_train = self.graphs[param_dict['dataset']]['train_directed']
        else:
            if 'neighborhood_mean' in param_dict and param_dict['neighborhood_mean']==True:
                graph_train = self.graphs[param_dict['dataset']]['train_mean']
            else:
                graph_train = self.graphs[param_dict['dataset']]['train']


        embedding_tester = model_tester( 
            parameters = model_params,
            seed = param_dict['seed'],
            quantization = param_dict['quantization'],
            apply_sigmoid = param_dict['apply_sigmoid'],
            n_embed_bits = param_dict['quantized_dim'],
            mapping = param_dict['mapping'],
            graph_train = graph_train)

        self.models[label] = embedding_tester
        return embedding_tester

    def load_observable_rankings(self,dataset):
        paths_ranks = {'KG_233':'/ceph/vinith/kg/embedding/experiments/observables/KG_233_ranks.pkl',
                'Wordnet':'/ceph/vinith/kg/embedding/experiments/observables/Wordnet_ranks.pkl',
                'Slashdot':'/ceph/vinith/kg/embedding/experiments/observables/Slashdot_ranks.pkl'}
        self.observable_test_ranks[dataset] = pickle.load(open(paths_ranks[dataset],'r'))
        return
    def load_observable_samples(self,dataset):
        paths_samples = {'KG_233':'/ceph/vinith/kg/embedding/experiments/observables/KG_233_samples.pkl',
                'Wordnet':'/ceph/vinith/kg/embedding/experiments/observables/Wordnet_samples.pkl',
                'Slashdot':'/ceph/vinith/kg/embedding/experiments/observables/Slashdot_samples.pkl'}
        self.observable_test_samples[dataset] = pickle.load(open(paths_samples[dataset],'r'))
        return
    def load_observable_scores(self,dataset):
        paths_scores = {'KG_233':'/ceph/vinith/kg/embedding/experiments/observables/KG_233_scores.pkl',
                'Wordnet':'/ceph/vinith/kg/embedding/experiments/observables/Wordnet_scores.pkl',
                'Slashdot':'/ceph/vinith/kg/embedding/experiments/observables/Slashdot_scores.pkl'}
        self.observable_test_scores[dataset] = pickle.load(open(paths_scores[dataset],'r'))
        return

    '''
    loads both train and test datasets given seed parameter
    '''
    def load_dataset(self,dataset_name):
        #hardcoded paths to datasets
        paths = {}
        paths['KG'] = os.path.join(BASE_DIR,'data/traindata_db233_minmentions10_minentity3.pkl')
        paths['Wordnet'] = os.path.join(BASE_DIR,'data/noun_relations.pkl')
        paths['Slashdot'] = os.path.join(BASE_DIR,'data/slashdot.pkl')
        paths['Flickr'] = os.path.join(BASE_DIR,'data/flickr.pkl')
        paths['Blogcatalog'] = os.path.join(BASE_DIR,'data/blogcatalog.pkl')

        if dataset_name.startswith('KG'):
            data,entity_to_idx = pickle.load(open(paths['KG'],'r'))
                
        elif dataset_name=='Wordnet':
            data,idx_to_entity,entity_to_idx = pickle.load(open(paths[dataset_name],'r'))

        elif dataset_name in ['Slashdot','Flickr','Blogcatalog']:
            data = pickle.load(open(paths[dataset_name],'r'))

        randstate = np.random.RandomState(self.seed)
        data = randstate.permutation(data)
        
        data_train,data_test = train_test_split(data,test_size=VAL_FRAC,random_state=self.seed)

        self.datasets[dataset_name] = {'train':data_train, 'test':data_test,'all':data}
        for label,dat in self.datasets[dataset_name].items():
            self.datasets[dataset_name][label+'_sym'] = np.concatenate([dat,dat[:,::-1]],axis=0)

        ##Now load the graphs for each of the three subsets (train, test, all)
        n_entities = max(data.flatten())+1
        self.graphs[dataset_name] = {'train':get_graph(data_train,n_entities),
                                'test': get_graph(data_test, n_entities),
                                'all' : get_graph(data, n_entities),
                                'train_directed':get_graph_directed(data_train,n_entities),
                                'test_directed': get_graph_directed(data_test, n_entities),
                                'all_directed' : get_graph_directed(data, n_entities),
                                'train_directed_mean':get_graph_directed_mean(data_train, n_entities),
                                'train_mean':get_graph_mean(data_train, n_entities)}


'''
finds precisions for a given set of recalls to check against, and for a given set of rankings
(used for observable-PR curves)
'''
def get_PR_observable(rankings,recalls):
    precisions_interpolated = 0
    for i,ranking in enumerate(rankings):
        ####AWKWARD HACK TO FIX REVERSAL BUG IN THE WAY MODEL_OBSERVABLES IS SAVING THE RANKINGS... SIGH.
        ranking = ranking[::-1]
        recall_idxs = np.ceil(recalls*len(ranking)).astype(int)-1
        precisions = (float(1)/ranking[recall_idxs])*(recall_idxs+1)
        precisions_interpolated += np.array([max(precisions[j:]) for j in range(len(precisions))])
    return (precisions_interpolated/len(rankings))

'''
finds precisions for a given set of recalls to check against.
'''
def get_PR_testset(emb,graph_test,recalls,n_samples=100,inversion=False,mapping='Hamming'):
    #numpy broadcast magics to compute the outer "product" distances between the sampled embs and the emb matrix
    #samples = np.random.randint(emb.shape[0],size=n_samples)
    randstate = np.random.RandomState(2052016)
    samples = np.unique(randstate.choice(graphs['test'].indices,size=n_samples)) #sample randomly from points that appear in a left-side of a relation in the test graph
    distances = np.zeros((len(samples),emb.shape[0]))
    if mapping == 'L2':
        metric = lambda vec,mat: np.sqrt(np.mean((vec-mat)**2,axis=1))
    else:
        metric = lambda vec,mat: np.mean(np.abs(vec-mat),axis=1)

    for i,sample in enumerate(samples):
        distances[i,:] = metric(emb[sample],emb)

    if inversion:
        distances = 1 - distances
    
    indptr_test = graphs['test'].indptr
    precisions_interpolated = 0
    for i,sample in enumerate(samples):
        #train_connected = graphs['train'].indices[indptr_train[sample]:indptr_train[sample+1]]
        test_connected = graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]
        #1. get the distances to the relevant documents
        counts = np.sum(np.sort(distances[i,test_connected])[:,np.newaxis]>=distances[i],axis=1)

        recall_idxs = np.ceil(recalls*len(test_connected)).astype(int)-1
        precisions = (float(1)/counts[recall_idxs])*(recall_idxs+1)
        precisions_interpolated += np.array([max(precisions[j:]) for j in range(len(precisions))])

    return (precisions_interpolated/len(samples))

'''
finds precisions for a given set of recalls to check against USING OBSERVABLES, AFTER PRERANKING WITH EMBEDDINGS
'''
def get_PR_observable_prerank(observable_scores,samples,prerank_type,emb,graphs,recalls,inversion=False,mapping='Hamming'):
    #first, extract parameters
    prerank_threshold = int(prerank_type.rsplit(':')[1])
    prerank_type = prerank_type.rsplit(':')[0]

    #first, get distances
#    emb_sampled = emb[samples]
#    emb_sampled = emb_sampled[:,np.newaxis,:]
    distances = np.zeros((len(samples),emb.shape[0]))
    if mapping == 'L2':
        metric = lambda vec,mat: np.sqrt(np.mean((vec-mat)**2,axis=1))
    else:
        metric = lambda vec,mat: np.mean(np.abs(vec-mat),axis=1)

    for i,sample in enumerate(samples):
        distances[i,:] = metric(emb[sample],emb)

    if inversion:
        distances = 1 - distances

    #second, find set to be reranked
    if prerank_type=='Observable_prerank_ball':
        prerank_filter = distances <= float(prerank_threshold)/emb.shape[1]
        sorted_idx = np.argsort(distances,axis=1)
        for i,sample in enumerate(samples):
            prerank_filter[i,sorted_idx[i,MAX_N:]] = False
        print np.sum(prerank_filter,axis=1)[:20]
        sys.stdout.flush()
        if np.any(prerank_filter):
            distances[prerank_filter] = -observable_scores[prerank_filter]
    elif prerank_type =='Observable_prerank_top':
        sorted_idx = np.argsort(distances,axis=1)
        for i,sample in enumerate(samples):
            distances[i,sorted_idx[i,:prerank_threshold]] = -observable_scores[i,sorted_idx[i,:prerank_threshold]]

    #and now can compute P/R using the same old machinery
    indptr_test = graphs['test'].indptr
    precisions_interpolated = 0
    for i,sample in enumerate(samples):
        #train_connected = graphs['train'].indices[indptr_train[sample]:indptr_train[sample+1]]
        test_connected = graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]
        #1. get the distances to the relevant documents
        counts = np.sum(np.sort(distances[i,test_connected])[:,np.newaxis]>=distances[i],axis=1)

        recall_idxs = np.ceil(recalls*len(test_connected)).astype(int)-1
        precisions = (float(1)/counts[recall_idxs])*(recall_idxs+1)
        precisions_interpolated += np.array([max(precisions[j:]) for j in range(len(precisions))])

    return (precisions_interpolated/len(samples))


'''
finds mean average precision for a random sampled set of queries 
'''
def get_MAP_testset(emb,graphs,n_samples=100,inversion=False,mapping='Hamming'):
    randstate = np.random.RandomState(2052016)
    samples = np.unique(randstate.choice(graphs['test'].indices,size=n_samples)) #sample randomly from points that appear in a left-side of a relation in the test graph
    distances = np.zeros((len(samples),emb.shape[0]))
    if mapping == 'L2':
        metric = lambda vec,mat: np.sqrt(np.mean((vec-mat)**2,axis=1))
    else:
        metric = lambda vec,mat: np.mean(np.abs(vec-mat),axis=1)

    for i,sample in enumerate(samples):
        distances[i,:] = metric(emb[sample],emb)

    if inversion:
        distances = 1 - distances
    
    indptr_test = graphs['test'].indptr
    MAP = []
    for i,sample in enumerate(samples):
        test_connected = graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]

        counts = np.sum(np.sort(distances[i,test_connected])[:,np.newaxis]>=distances[i],axis=1)
        precisions = (float(1)/counts)*np.arange(1,len(counts)+1)
        #precisions_interpolated = np.array([max(precisions[j:] for j in range(len(precisions)))])
        MAP.append(np.mean(precisions))

    return np.mean(MAP)

def get_MAP_observable(rankings):
    MAP = []
    for ranks in rankings:
        ####AWKWARD HACK TO FIX REVERSAL BUG IN THE WAY MODEL_OBSERVABLES IS SAVING THE RANKINGS... SIGH.
        ranks = ranks[::-1]
        precisions = (float(1)/ranks)*np.arange(1,len(ranks)+1)
        MAP.append(np.mean(precisions))
    return np.mean(MAP)

def get_MAP_unigram(data_train,graphs,n_samples):
    randstate = np.random.RandomState(2052016)
    unigram_sampler = neg_sampling.UnigramSampler(data_train,power=1,laplace = 0,randstate=randstate)
    
    left_samples = np.unique(randstate.choice(graphs['test'].indices,size=n_samples))
    distances = -unigram_sampler.pdf #smallest distance is highest unigram probability

    indptr_test = graphs['test'].indptr
    MAP = []
    for i,sample in enumerate(left_samples):
        test_connected = graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]

        counts = np.sum(np.sort(distances[test_connected])[:,np.newaxis]>=distances,axis=1)
        precisions = (float(1)/counts)*np.arange(1,len(counts)+1)
        #precisions_interpolated = np.array([max(precisions[j:] for j in range(len(precisions)))])
        MAP.append(np.mean(precisions))

    return np.mean(MAP)

def get_MAP_observable_prerank(observable_scores,samples,prerank_type,emb,graphs,n_samples=100,inversion=False,mapping='Hamming'):
    #first, extract parameters
    prerank_threshold = int(prerank_type.rsplit(':')[1])
    prerank_type = prerank_type.rsplit(':')[0]

    #first, get distances
    distances = np.zeros((len(samples),emb.shape[0]))
    if mapping == 'L2':
        metric = lambda vec,mat: np.sqrt(np.mean((vec-mat)**2,axis=1))
    else:
        metric = lambda vec,mat: np.mean(np.abs(vec-mat),axis=1)

    for i,sample in enumerate(samples):
        distances[i,:] = metric(emb[sample],emb)

    if inversion:
        distances = 1 - distances

    #second, find set to be reranked
    if prerank_type=='Observable_prerank_ball':
        prerank_filter = distances <= float(prerank_threshold)/emb.shape[1]
        sorted_idx = np.argsort(distances,axis=1)
        for i,sample in enumerate(samples):
            prerank_filter[i,sorted_idx[i,MAX_N:]] = False

        print np.sum(prerank_filter,axis=1)[:20]
        sys.stdout.flush()

        if np.any(prerank_filter):
            distances[prerank_filter] = -observable_scores[prerank_filter]
    elif prerank_type =='Observable_prerank_top':
        sorted_idx = np.argsort(distances,axis=1)
        for i,sample in enumerate(samples):
            distances[i,sorted_idx[i,:prerank_threshold]] = -observable_scores[i,sorted_idx[i,:prerank_threshold]]

    indptr_test = graphs['test'].indptr
    MAP = []
    for i,sample in enumerate(samples):
        test_connected = graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]
        counts = np.sum(np.sort(distances[i,test_connected])[:,np.newaxis]>=distances[i],axis=1)
        precisions = (float(1)/counts)*np.arange(1,len(counts)+1)
        MAP.append(np.mean(precisions))
    return np.mean(MAP)


def get_PR_allset(emb,graphs,radius):
    raise NotImplementedError


if __name__=='__main__':
    params = {'NCE1': 
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE1',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
              'NCE2': 
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE2',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'NCE3': 
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE3',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'NCE4': 
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE4',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'NCE5': 
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
             'KG1':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG1',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG2':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG2',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG2b':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG2b',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG2c':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG2c',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG3':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG3',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG6': 
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG6',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'KG7': 
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG7',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'KG8': 
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG8',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'Slashdot1': 
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT1',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'Slashdot2': 
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT2',
                'quantization':'Sample_deterministic',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100  
                },
                'KG4':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG4',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                'KG5':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG5',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 100
                },
                }
    params = {'NCE0':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE0',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE0.5':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE0.5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_KG0':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG0',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_KG0.5':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG0.5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT0':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT0',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT0.5':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT0.5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                }
             }
    params = {'NCE6':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE6',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE7':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE0.5',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE8':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE8',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE9':
                {
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE9',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_KG9':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG9',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_KG10':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG10',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT3':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT3',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT4':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT4',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT5':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT6':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT6',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT7':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT7',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT8':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT8',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                }
    }
    params = {'NCE_KG9':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG9',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_KG10':
                {
                'dataset':'KG_233',
                'fiedler':None,
                'exp':'NCE_KG10',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT3':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT3',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT4':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT4',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT5':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT5',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT6':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT6',
                'quantization':'Sample',
                'quantized_dim':25,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE_SLASHDOT8':
                {
                'dataset':'Slashdot',
                'fiedler':None,
                'exp':'NCE_SLASHDOT8',
                'quantization':'Sample',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                }
    }
    keys = params.keys()
    for label in keys:
        params[label]['reranking'] = 'Embedding_rank' #'Observable_prerank_ball:1' #
        params[label]['n_precision_samples'] = 1000
        params[label]['best_sampled'] = True
        params[label]['quantization'] = 'Sample_deterministic'
        params[label+'_rerank'] = copy.deepcopy(params[label])
        if params[label]['quantized_dim'] == 10:
            params[label+'_rerank']['reranking'] = 'Observable_prerank_ball:1'
        elif params[label]['quantized_dim'] == 25:
            params[label+'_rerank']['reranking'] = 'Observable_prerank_ball:3'
        else:
            print "error"
        # params[label+'_det'] = copy.deepcopy(params[label])
        # params[label+'_det']['quantization'] = 'Sample_deterministic'
    # params = {'KG_unigram': {
    #             'reranking':'Unigram',
    #             'dataset':'KG_233',
    #             'fiedler':None,
    #             'exp':'S38',
    #             'quantization':'Sample',
    #             'quantized_dim':10,
    #             'apply_sigmoid':True,
    #             'mapping':'Hamming',
    #             'loss':'MAP',
    #             'loss_set':'test',
    #             'P/R points':None,
    #             'seed':2052016,
    #             'n_precision_samples': 1000,
    #             'best_sampled': True
    #             },
    #            'Wordnet_unigram': {
    #             'reranking':'Unigram',
    #             'dataset':'Wordnet',
    #             'fiedler':None,
    #             'exp':'DG14',
    #             'quantization':'Sample',
    #             'quantized_dim':10,
    #             'apply_sigmoid':True,
    #             'mapping':'Hamming',
    #             'loss':'MAP',
    #             'loss_set':'test',
    #             'P/R points':None,
    #             'seed':2052016,
    #             'n_precision_samples': 1000,
    #             'best_sampled': True
    #             },
    #            'Slashdot_unigram': {
    #             'reranking':'Unigram',
    #             'dataset':'Slashdot',
    #             'fiedler':None,
    #             'exp':'DG25',
    #             'quantization':'Sample',
    #             'quantized_dim':10,
    #             'apply_sigmoid':True,
    #             'mapping':'Hamming',
    #             'loss':'MAP',
    #             'loss_set':'test',
    #             'P/R points':None,
    #             'seed':2052016,
    #             'n_precision_samples': 1000,
    #             'best_sampled': True
    #             }}

    X = results(params)
    print X.df.sort('exp')
    params = {'NCE7':
                {
                'reranking':'Embedding_rank',
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE7',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
             'NCE7 reranked':
                {
                'reranking':'Observable_prerank_ball:1',
                'dataset':'Wordnet',
                'fiedler':None,
                'exp':'NCE7',
                'quantization':'Sample_deterministic',
                'quantized_dim':10,
                'apply_sigmoid':True,
                'mapping':'Hamming',
                'loss':'MAP',
                'loss_set':'test',
                'P/R points':None,
                'seed':2052016,
                'n_precision_samples': 1000
                },
    }
    params = {
             'Original':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-1':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-1',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-1b':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-1b',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-1c':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-1c',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-1d':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-1d',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-2b':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-2b',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-2c':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-2c',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-3':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-3',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4a',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4a2':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4a2',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4c':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4c',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4d':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4d',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              '14-4e':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14-4e',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
    }
    params = {
             'DG14':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG14',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              'DG164':
              {
              'dataset':'Wordnet',
              'fiedler':None,
              'exp':'DG164',
              'quantization':'Sample',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              'DG25':
              {
              'dataset':'Slashdot',
              'fiedler':None,
              'exp':'DG25',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              'DG30b':
              {
              'dataset':'Slashdot',
              'fiedler':None,
              'exp':'DG30b',
              'quantization':'Sample',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              'S36':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'exp':'S36',
              'quantization':'Sample',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
              'S38':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'exp':'S38',
              'quantization':'Sample',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'n_precision_samples': 100
              },
    }
    # params = {
    #         '14-1 1':
    #           {
    #           'dataset':'Wordnet',
    #           'fiedler':None,
    #           'exp':'DG14-1',
    #           'quantization':'Sample',
    #           'quantized_dim':10,
    #           'apply_sigmoid':True,
    #           'mapping':'Hamming',
    #           'loss':'MAP',
    #           'loss_set':'test',
    #           'P/R points':None,
    #           'seed':2052016,
    #           'n_precision_samples': 100
    #           },
    #           '14-1 2':
    #           {
    #           'dataset':'Wordnet',
    #           'fiedler':None,
    #           'exp':'DG14-1',
    #           'quantization':'Sample_deterministic',
    #           'quantized_dim':10,
    #           'apply_sigmoid':True,
    #           'mapping':'Hamming',
    #           'loss':'MAP',
    #           'loss_set':'test',
    #           'P/R points':None,
    #           'seed':2052016,
    #           'n_precision_samples': 100
    #           },
    # }

    # import models.model_neighborhood as model_neighborhood
    # hypers = get_hypers('3NB4')
    # n_entities = max(X.datasets['KG_233']['all'].flatten())+1
    # Y = model_neighborhood.simple_linkpred(dim=hypers['DIM'],
    #                  n_entities=n_entities,
    #                  batch_size=hypers['BATCH_SIZE'],
    #                  n_samples=hypers['N_SAMPLES'],
    #                  embedding_type=hypers['EMBEDDING_TYPE'],
    #                  parameterization=hypers['PARAMETERIZATION'],
    #                  init_a = hypers['INIT_A'],
    #                  init_b = hypers['INIT_B'],
    #                  init_a_n = hypers['INIT_A_N'],
    #                  objective_samples=hypers['OBJECTIVE_SAMPLES'],
    #                  neighborhood = hypers['NEIGHBORHOOD'],
    #                  transform_scaling = hypers['TRANSFORM_SCALING'],
    #                  seed = hypers['SEED'],
    #                  quantile_floor_and_ceiling = hypers['QUANTILE_FLOOR_AND_CEILING'],
    #                  neighborhood_mean = hypers['NEIGHBORHOOD_MEAN'],
    #                  graph_symmetry = hypers['GRAPH_SYMMETRY'])
    # for xparam,yparam in zip(X.models['Testing'].parameters,Y.params):
    #     yparam.set_value(xparam)

    # #can we replicate the validation sampling +expectation log loss from the training? sanity check...
    # train_bundle,val_bundle = Y.get_training_fn(idxs_train.astype(np.int32),idxs_test.astype(np.int32),training='ADAGRAD',debug='None',training_param=0.9)
    # print "validation from model_neighborhood: ", val_bundle[1]()
    # #In [4]: val_bundle[1]()
    # #Out[4]: [array(0.7869485813711169), array(0.7952776816146704)]
    # #ding ding ding

    # #compare specific embeddings in model_neighborhood (Y) with same embeddings in model_testing (X)
    # print X.models['Testing'].binary_embeddings[Y.x1_idxs.get_value()[0]]
    # print Y.get_embed(Y.x1_idxs,None,Y.x1_batch_neighborhood).eval()[0]
    # print Y.x1_batch_neighborhood.get_value()[0]

    # pos_loss_X = -T.mean(T.log(X.models['Testing'].pos_probs))
    # data_test = X.datasets['KG_233']['test']
    # #compare positive losses
    # print pos_loss_X.eval({X.models['Testing'].x1_idxs:np.transpose(data_test).astype(np.int32)[0],
    #                  X.models['Testing'].x2_idxs:np.transpose(data_test).astype(np.int32)[1]})
    # print Y.pos_losses[0].eval()

    # #compare negative losses
    # neg_loss_X = -T.mean(T.log(1.0 - X.models['Testing'].neg_probs))
    # print neg_loss_X.eval({X.models['Testing'].x1_idxs:np.transpose(data_test).astype(np.int32)[0],
    #                  X.models['Testing'].x2_idxs:np.transpose(data_test).astype(np.int32)[1]})
    # print Y.neg_losses[0].eval()

    # #compare negative embeddings
    # print Y.x1_idxs_negative.eval()[:100]
    # #[   981 111996  16561  25042  38173  60140  72532    458 109617  68652]
    # #second time, same random seed for theano_rng
    # #[   981 111996  16561  25042  38173  60140  72532    458 109617  68652]
    # #third time, different random seed:
    # #[ 38464  18510 109182  59613  31791  15065 112585  27718 108204  41636]
    # print Y.x1_idxs.get_value()[:100]
    

    # print Y.x1_idxs.eval()[:10]
    # print Y.x1neg_emb.eval()[2]
    # print X.models['Testing'].binary_embeddings[Y.x1_idxs.get_value()[2]]

        #print x1_neg_fromX[:10]
    #[106617 104156  16561  25042  81429   7724  72532 103605 109617 101200]
    
   
    # #Y-model_neighborhood can have the negative values be set before evaluating negative loss.
    # Y.x1_idxs_negative.set_value(x1_neg_fromX.astype(np.int32))
    # Y.x2_idxs_negative.set_value(x2_neg_fromX.astype(np.int32))
    # print Y.neg_losses[0].eval()
    # #0.455274444401 ---> thumbs up.

    # print Y.x1neg_emb.eval()[0]
    # print Y.x2neg_emb.eval()[0]
    # x1_negemb_fromX = X.models['Testing'].x1neg_emb.eval({X.models['Testing'].x1_idxs:np.transpose(data_test).astype(np.int32)[0]})
    # x2_negemb_fromX = X.models['Testing'].x2neg_emb.eval({X.models['Testing'].x1_idxs:np.transpose(data_test).astype(np.int32)[0], 
    #                                                           X.models['Testing'].x2_idxs:np.transpose(data_test).astype(np.int32)[1]})
    # print x1_negemb_fromX[0]
    # print x2_negemb_fromX[0]
    




    # #test whether they evaluate to the same ON THE SAME POSITIVE AND NEGATIVE SAMPLES
    # #start by determining common inputs

    # #data
    # data_test = X.datasets['KG_233']['test']
    # data_train = X.datasets['KG_233']['train']
    # pdb.set_trace()

    
    # #positives
    # x1_idxs = np.transpose(data_test).astype(np.int32)[0]
    # x2_idxs = np.transpose(data_test).astype(np.int32)[1]

    # #negatives
    # x1_idxs_negative = x1_idxs.copy()
    # x2_idxs_negative = x2_idxs.copy()
    # idxs_neg_idx = np.random.permutation(range(len(x1_idxs)))
    # x1_idxs_negative[idxs_neg_idx[:len(idxs_neg_idx)/2]] = np.random.randint(n_entities,size=len(idxs_neg_idx)/2)
    # x2_idxs_negative[idxs_neg_idx[len(idxs_neg_idx)/2:]] = np.random.randint(n_entities,size=len(idxs_neg_idx[len(idxs_neg_idx)/2:]))
    # # x1_idxs_negative_fromX = X.models['Testing'].x1_idxs_negative.eval({X.models['Testing'].x1_idxs:x1_idxs})
    # # x2_idxs_negative_fromX = X.models['Testing'].x2_idxs_negative.eval({X.models['Testing'].x1_idxs:x1_idxs,
    #                                                                     # X.models['Testing'].x2_idxs:x2_idxs})
    
    # #neighborhoods    
    # graph_train = X.models['Testing'].graph_train
    # x1_neighborhood = graph_train[x1_idxs]
    # x2_neighborhood = graph_train[x2_idxs]
    # x1neg_neighborhood = graph_train[x1_idxs_negative]
    # x2neg_neighborhood = graph_train[x2_idxs_negative]
    # # x1neg_neighborhood = graph_train[x1_idxs_negative_fromX]
    # # x2neg_neighborhood = graph_train[x2_idxs_negative_fromX]

    # #set values for X
    # X.models['Testing'].x1_idxs.set_value(x1_idxs)
    # X.models['Testing'].x2_idxs.set_value(x2_idxs)
    # X.models['Testing'].x1_idxs_negative.set_value(x1_idxs_negative)
    # X.models['Testing'].x2_idxs_negative.set_value(x2_idxs_negative)

    # #set values for Y    
    # Y.x1_idxs.set_value(x1_idxs.astype(np.int32))
    # Y.x2_idxs.set_value(x2_idxs.astype(np.int32))
    # Y.x1_idxs_negative.set_value(x1_idxs_negative.astype(np.int32))
    # Y.x2_idxs_negative.set_value(x2_idxs_negative.astype(np.int32))
    # # Y.x1_idxs_negative.set_value(x1_idxs_negative_fromX.astype(np.int32))
    # # Y.x2_idxs_negative.set_value(x2_idxs_negative_fromX.astype(np.int32))
    # Y.x1_batch_neighborhood.set_value(x1_neighborhood)
    # Y.x2_batch_neighborhood.set_value(x2_neighborhood)
    # Y.x1neg_batch_neighborhood.set_value(x1neg_neighborhood)
    # Y.x2neg_batch_neighborhood.set_value(x2neg_neighborhood)

    # #evaluate loss for X
    # neg_loss_X = -T.mean(T.log(1.0 - X.models['Testing'].neg_probs))
    # pos_loss_X = -T.mean(T.log(X.models['Testing'].pos_probs))
    # print neg_loss_X.eval()
    # print pos_loss_X.eval()

    # #evaluate loss for Y
    # print Y.neg_losses[0].eval()
    # print Y.pos_losses[0].eval()

    # #match on both positives and negatives
    # #catch: negatives are not what is being claimed independently by Y's validate function.







    # #Try the same in the reverse direction: values from Y's generator into X's machinery

    # #1. Y's validation output with its own values of idxs in
    # train_bundle,val_bundle = Y.get_training_fn(data_train.astype(np.int32),data_test.astype(np.int32),training='ADAGRAD',debug='None',training_param=0.9)
    # print "validation from model_neighborhood: ", val_bundle[1]()
    # #validation from model_neighborhood:  [array(0.4117507949758973), array(0.5617929410525468)]

    # #2. get common values out of Y
    # x1_idxs = Y.x1_idxs.get_value()
    # x2_idxs = Y.x2_idxs.get_value()
    # x1_idxs_negative = Y.x1_idxs_negative.get_value()
    # x2_idxs_negative = Y.x2_idxs_negative.get_value()

    # #3. set values for X
    # X.models['Testing'].x1_idxs.set_value(x1_idxs)
    # X.models['Testing'].x2_idxs.set_value(x2_idxs)
    # X.models['Testing'].x1_idxs_negative.set_value(x1_idxs_negative)
    # X.models['Testing'].x2_idxs_negative.set_value(x2_idxs_negative)

    # #evaluate loss for X
    # neg_loss_X = -T.mean(T.log(1.0 - X.models['Testing'].neg_probs))
    # pos_loss_X = -T.mean(T.log(X.models['Testing'].pos_probs))
    # print neg_loss_X.eval()
    # print pos_loss_X.eval()

    # #and Y
    # print Y.neg_losses[0].eval()
    # print Y.pos_losses[0].eval()

    # #result: agree on positive, disaggree on negative (Dramatically!)
    # #how about negative embeddings?
    # print Y.get_embed(Y.x1_idxs_negative,None,Y.x1neg_batch_neighborhood).eval()[0]
    # print X.models['Testing'].x1neg_emb.eval()[0]
    # print Y.get_embed(Y.x2_idxs_negative,None,Y.x2neg_batch_neighborhood).eval()[0]
    # print X.models['Testing'].x2neg_emb.eval()[0]

    # #They disagree on negative embeddings (in this specific case, on x2)! Track/replicate the actual embedding creation process...
    # import theano.sparse
    # nbhood_emb_Y = theano.sparse.structured_dot(Y.x2neg_batch_neighborhood,Y.emb).eval()[0]

    # graph_train = X.models['Testing'].graph_train
    # raw_embeddings_X = X.models['Testing'].parameters[0]
    # nbhood_emb_X = graph_train.dot(raw_embeddings_X)[x2_idxs_negative[0]]

    # print nbhood_emb_Y
    # print nbhood_emb_X
    # #already off!!


    # #Try even simpler test
    # batch_neighborhood_Y = Y.x2neg_batch_neighborhood.get_value()[0]
    # batch_neighborhood_X = graph_train[x2_idxs_negative[0]]
    # print batch_neighborhood_Y
    # print batch_neighborhood_X
    # #disagree even on batch neighborhood!

    # #even simpler: confirm that they are looking at the same list of x2neg idxs....
    # print Y.x2_idxs_negative.eval()[0], X.models['Testing'].x2_idxs_negative.eval()[0], x2_idxs_negative[0]
    # #all the same

    # #verify that the graphs are the same...
    # print Y.graph[0]
    # print X.models['Testing'].graph_train[0]
    # #both the same....


    #given the above, batch neighborhood in Y is seemingly inconsistent...
    #So let's replicate the creation of batch neighborhood

    #####FOUND THE BUG.











    # nbhood_embed = X.graphs['KG_233']['train_directed_mean'][38464].dot(X.models['Testing'].parameters[0]).dot(X.models['Testing'].parameters[-2])*X.models['Testing'].parameters[-1]+X.models['Testing'].parameters[0][38464]
    # print nbhood_embed


    '''
params = {
             'Observable':
              {
              'dataset':'Wordnet',
              'fiedler':['unnormalized',1000],
              'reranking':'Embedding_rank',
              'exp':None,
              'quantization':'Raw',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'L2',
              'loss':'MAP',
              'loss_set':'test',
              'P/R points':np.arange(1,9)*.1,
              'seed':2052016
              }
             }
    X = results(params)
    print X.df

    'Observable':
              {
              'dataset':'KG_233',
              'reranking':'Observable_prerank_top:1000',
              'exp':'DGb10',
              'quantization':'Raw',
              'quantized_dim':100,
              'apply_sigmoid':True,
              'mapping':'L2',
              'loss':'P/R',
              'loss_set':'test',
              'P/R points':np.arange(1,9)*.1,
              'seed':2052016
         exp       label  precision  recall   Name val loss expected  \
0  DGb10  Observable   0.124178     0.1  DGb10         0.5381437
1  DGb10  Observable   0.110693     0.2  DGb10         0.5381437
2  DGb10  Observable   0.097694     0.3  DGb10         0.5381437
3  DGb10  Observable   0.088297     0.4  DGb10         0.5381437
4  DGb10  Observable   0.081246     0.5  DGb10         0.5381437
5  DGb10  Observable   0.061180     0.6  DGb10         0.5381437
6  DGb10  Observable   0.052166     0.7  DGb10         0.5381437
7  DGb10  Observable   0.046458     0.8  DGb10         0.5381437
    'dataset':'KG_233',
              'reranking':'Embedding_rank',
              'exp':'DGb10',
              'quantization':'Raw',
              'quantized_dim':100,
              'apply_sigmoid':True,
              'mapping':'L2',
              'loss':'P/R',
              'loss_set':'test',
              'P/R points':np.ar
0  DGb10  Observable   0.148144     0.1  DGb10         0.5381437
1  DGb10  Observable   0.118059     0.2  DGb10         0.5381437
2  DGb10  Observable   0.092121     0.3  DGb10         0.5381437
3  DGb10  Observable   0.078824     0.4  DGb10         0.5381437
4  DGb10  Observable   0.073432     0.5  DGb10         0.5381437
5  DGb10  Observable   0.049183     0.6  DGb10         0.5381437
6  DGb10  Observable   0.044476     0.7  DGb10         0.5381437
7  DGb10  Observable   0.040453     0.8  DGb10         0.5381437

  val loss sampled  dim dataset emb type     sampling
0         1.103739  100  KG_233     real  expectation
1         1.103739  100  KG_233     real  expectation
2         1.103739  100  KG_233     real  expectation
3         1.103739  100  KG_233     real  expectation
4         1.103739  100  KG_233     real  expectation
5         1.103739  100  KG_233     real  expectation
6         1.103739  100  KG_233     real  expectation
7         1.103739  100  KG_233     real  expectation


    'dataset':'KG_233',
              'reranking':'Observable_prerank_top:1000000',
              'exp':'DGb10',
              'quantization':'Raw',
              'quantized_dim':100,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'P/R',
              'loss_set':'test',
              'P/R points':np.arange(1,9)*.1,
              'seed':2052016
         exp       label  precision  recall   Name val loss expected  \
0  DGb10  Observable   0.135339     0.1  DGb10         0.5381437
1  DGb10  Observable   0.121724     0.2  DGb10         0.5381437
2  DGb10  Observable   0.107283     0.3  DGb10         0.5381437
3  DGb10  Observable   0.098696     0.4  DGb10         0.5381437
4  DGb10  Observable   0.092348     0.5  DGb10         0.5381437
5  DGb10  Observable   0.071758     0.6  DGb10         0.5381437
6  DGb10  Observable   0.062983     0.7  DGb10         0.5381437
7  DGb10  Observable   0.056740     0.8  DGb10         0.5381437

  val loss sampled  dim dataset emb type     sampling
0         1.103739  100  KG_233     real  expectation
1         1.103739  100  KG_233     real  expectation
2         1.103739  100  KG_233     real  expectation
3         1.103739  100  KG_233     real  expectation
4         1.103739  100  KG_233     real  expectation
5         1.103739  100  KG_233     real  expectation
6         1.103739  100  KG_233     real  expectation
7         1.103739  100  KG_233     real  expectation

:)''' 

    #test observable ranking
    # params = {
    #          'Observable':
    #           {
    #           'dataset':'KG_233',
    #           'reranking':'Observable_rank',
    #           'loss':'P/R',
    #           'loss_set':'test',
    #           'P/R points':np.arange(1,9)*.1,
    #           'seed':2052016
    #           }
    #          }
    '''
            label  precision  recall
0  Observable   0.135339     0.1
1  Observable   0.121724     0.2
2  Observable   0.107283     0.3
3  Observable   0.098696     0.4
4  Observable   0.092348     0.5
5  Observable   0.071758     0.6
6  Observable   0.062983     0.7
7  Observable   0.056740     0.8'''

'''
params = {
             'Testing_1':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'S36',
              'quantization':'Sample',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_2':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DGb10',
              'quantization':'ITQ',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_3':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW14',
              'quantization':'ITQ',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_4':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW12',
              'quantization':'ITQ',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_5':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW12',
              'quantization':'Raw',
              'quantized_dim':25,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_6':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW11',
              'quantization':'Raw',
              'quantized_dim':20,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_7':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW10',
              'quantization':'Raw',
              'quantized_dim':15,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
              'Testing_8':
              {
              'dataset':'KG_233',
              'fiedler':None,
              'reranking':'Embedding_rank',
              'exp':'DW9',
              'quantization':'Raw',
              'quantized_dim':10,
              'apply_sigmoid':True,
              'mapping':'Hamming',
              'loss':'Joint Entropy Plugin',
              'loss_set':'test',
              'P/R points':None,
              'seed':2052016,
              'neighborhood_mean':False,
              'graph_symmetry':False
              },
             }
'''