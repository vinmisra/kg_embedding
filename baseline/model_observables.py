import pdb,os
import scipy
import numpy as np
import pandas as pd
import cPickle as pickle
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import math
import sys

paths_datasets = {'KG_233':'/ceph/vinith/kg/embedding/data/traindata_db233_minmentions10_minentity3.pkl',
         'Wordnet':'/ceph/vinith/kg/embedding/data/noun_relations.pkl',
         'Slashdot':'/ceph/vinith/kg/embedding/data/slashdot.pkl',
         'Flickr':'/ceph/vinith/kg/embedding/data/flickr.pkl',
         'Blogcatalog':'/ceph/vinith/kg/embedding/data/blogcatalog.pkl'}

BASE_DUMP_PATH = '/ceph/vinith/kg/embedding/experiments/observables/'

paths_models = {}
paths_ranks = {}
paths_scores = {}
paths_samples = {}
paths_training_data = {}

for dataset_name in paths_datasets:
    paths_models[dataset_name] = os.path.join(BASE_DUMP_PATH,dataset_name+'_model.pkl')
    paths_ranks[dataset_name] = os.path.join(BASE_DUMP_PATH,dataset_name+'_ranks.pkl')
    paths_scores[dataset_name] = os.path.join(BASE_DUMP_PATH,dataset_name+'_scores.pkl')
    paths_samples[dataset_name] = os.path.join(BASE_DUMP_PATH,dataset_name+'_samples.pkl')
    paths_training_data[dataset_name] = os.path.join(BASE_DUMP_PATH,dataset_name+'_training_data.pkl')

SEED = 2052016

class model_observables(object):
    def __init__(self):
        return

    def load_results(self):
        self.ranks = pickle.load(open(paths_ranks[self.dataset],'r'))
        self.scores = pickle.load(open(paths_scores[self.dataset],'r'))
        self.samples = pickle.load(open(paths_samples[self.dataset],'r'))

    def dump_results(self):
        pickle.dump(self.ranks,open(paths_ranks[self.dataset],'w'))
        pickle.dump(self.scores,open(paths_scores[self.dataset],'w'))
        pickle.dump(self.samples,open(paths_samples[self.dataset],'w'))

    def load_model(self):
        self.model = pickle.load(open(paths_models[self.dataset],'r'))

    def dump_model(self):
        pickle.dump(self.model,open(paths_models[self.dataset],'w'))


    '''
    loads dataset with train/test splits, and loads corresponding graphs as well
    '''
    def load_dataset(self,dataset='KG_233',VALIDATION_HOLDOUT=0.05):
        self.dataset=dataset
        self.data = pickle.load(open(paths_datasets[self.dataset],'r'))
        if self.dataset == 'KG_233':
            self.data,self.entity_to_idx = self.data
            self.idx_to_entity = {}
            for entity,idx in self.entity_to_idx.items():
                self.idx_to_entity[idx] = entity

        elif self.dataset == 'Wordnet':
            self.data,self.idx_to_entity,self.entity_to_idx = self.data

        #shuffle and split into train and test
        randstate = np.random.RandomState(SEED)
        self.data = randstate.permutation(self.data)
        
        data_train,data_test = train_test_split(self.data,test_size=VALIDATION_HOLDOUT,random_state=SEED)
        self.data = {'train':data_train, 'test':data_test,'all':self.data}

        ##Now load the graphs for each of the three subsets (train, test, all)
        self.n_entities = max(self.data['all'].flatten())+1
        self.graphs = {'train':get_graph(self.data['train'],self.n_entities),
                                'test': get_graph(self.data['test'], self.n_entities),
                                'all' : get_graph(self.data['all'], self.n_entities)}


    def train(self):
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(self.features_train)

        self.model = LogisticRegression(C=10,penalty='l1')
        self.model.fit(features_scaled,self.labels_train.transpose())

    def gen_training_data(self,N_TRAINING_SAMPLES,dist_type='UNIFORM',use_cache = False):

        if os.path.exists(paths_training_data[self.dataset]) and use_cache:
            self.features_train,self.labels_train = pickle.load(open(paths_training_data[self.dataset],'r'))
            return

        #otherwise, we have to generate training data from scratch
        #generate positive samples
        self.train_data_positive = self.data['train'][:N_TRAINING_SAMPLES]

        #generate negative samples
        randstate = np.random.RandomState(SEED)
        if dist_type == 'UNIGRAM':
            randidxs = randstate.permutation(self.data['train'].flatten())[:N_TRAINING_SAMPLES] #unigram distribution
        elif dist_type == 'UNIFORM':
            randidxs = randstate.randint(self.n_entities,size=N_TRAINING_SAMPLES)#uniform distribution
        else:
            print 'UNRECOGNIZED DISTRIBUTION TYPE!!!'
        randchoice = randstate.binomial(n=1,p=.5,size=(N_TRAINING_SAMPLES))
        self.train_data_negative = self.train_data_positive.copy()

        self.train_data_negative[range(len(self.train_data_negative)),randchoice] = randidxs

        ###Featurization
        featurized_data_positive = featurize_samples(self.graphs['train'],self.train_data_positive)
        featurized_data_negative = featurize_samples(self.graphs['train'],self.train_data_negative)
        self.features_train = np.concatenate([featurized_data_positive,featurized_data_negative],axis=0)

        self.labels_train = np.concatenate([np.ones(len(featurized_data_positive)),0*np.ones(len(featurized_data_negative))])
        self.labels_train = self.labels_train.astype(np.int)

        pickle.dump((self.features_train,self.labels_train),open(paths_training_data[self.dataset],'w'))

    def gen_results(self,N_TEST_SAMPLES):

        #otherwise, generate it all from scratch
        #generate random samples for test data
        randstate = np.random.RandomState(SEED)
        self.samples = np.unique(randstate.choice(self.graphs['test'].indices,size=N_TEST_SAMPLES))

        #generate scores
        self.features_scaled = self.scaler.transform(featurize_all(self.graphs['train'],self.samples))
        self.scores = self.model.predict_proba(self.features_scaled)[:,1]
        self.scores = self.scores.reshape((len(self.samples),self.n_entities))

        #generate rankings for each test-set neighbor
        self.ranks = []
        indptr_test = self.graphs['test'].indptr
        for i,sample in enumerate(self.samples):
            test_connected = self.graphs['test'].indices[indptr_test[sample]:indptr_test[sample+1]]
            counts = np.sum(np.sort(self.scores[i,test_connected])[:,np.newaxis]<=self.scores[i],axis=1)
            self.ranks.append(counts)

    '''
    Returns a vector of scores, for every entity in the graph.
    Node can either be a string name of a node (entity_to_idx) or its idx.
    Assume that dataset is loaded.
    Assume that model is either trained or loaded.
    Assume that training data has been loaded (for purposes of scaling).
    '''
    def score_neighbors(self,node):
        if type(node)==type('string'):
            idx = self.entity_to_idx[node]
        elif type(node)==type(1):
            idx = node

        #featurize and normalize features
        features = featurize_all(self.graphs['all'],[idx])
        scaler = StandardScaler()
        scaler.fit(self.features_train)
        features_scaled = scaler.transform(features)

        #compute scores using model
        scores = self.model.predict_proba(features_scaled)[:,1]

        return scores






'''
defines graph sparse matrix given data matrix of the usual form
'''
def get_graph(data, n_entities):
    rows = np.transpose(data)[0]
    cols = np.transpose(data)[1]

    rows_sym = np.concatenate([rows,cols])
    cols_sym = np.concatenate([cols,rows])
    vals_sym = np.ones(2*len(rows))
    return csr_matrix((vals_sym,(rows_sym,cols_sym)),shape=(n_entities,n_entities))


def featurize_samples(graph,data):
    #generate raw features
    raw_features = []
    raw_features.append(get_common_neighbors(graph,data))
    raw_features.append(get_AA(graph,data))
    raw_features.append(get_AA(graph,data,lambda x: (1+x)**(-.5)))
    raw_features.append(get_AA(graph,data,lambda x: (1+x)**(-.3)))

    #perform transformations
    transformations = [lambda x: x,
                       lambda x: np.log(x+1),
                       lambda x: x**.5,
                       lambda x: x**.3,
                       lambda x: x**2]
    def transform_features(feature_list):
        output_features = []
        for transform in transformations:
            for feature in feature_list:
                output_features.append(transform(feature))
        return output_features

    transformed_features = np.vstack(transform_features(raw_features)).transpose()

    return transformed_features

def featurize_all(graph,samples):
    raw_features = []
    raw_features.append(get_common_neighbors_all(graph,samples))
    raw_features.append(get_AA_all(graph,samples))
    raw_features.append(get_AA_all(graph,samples,lambda x: (1+x)**(-.5)))
    raw_features.append(get_AA_all(graph,samples,lambda x: (1+x)**(-.3)))

    #perform transformations
    transformations = [lambda x: x,
                       lambda x: np.log(x+1),
                       lambda x: x**.5,
                       lambda x: x**.3,
                       lambda x: x**2]
    def transform_features(feature_list):
        output_features = []
        for transform in transformations:
            for feature in feature_list:
                output_features.append(transform(feature))
        return output_features

    transformed_features = np.vstack(transform_features(raw_features)).transpose()

    return transformed_features


def get_common_neighbors(graph_local, data_local):
    common_neighbors = np.zeros(len(data_local))

    for i,edge in enumerate(data_local):
        common_neighbors[i] = graph_local.getrow(edge[0]).dot(graph_local.getrow(edge[1]).transpose()).toarray()[0,0]
        
    sys.stdout.flush()
    return common_neighbors

def get_AA(graph_local, data_local, func=lambda x: 1/(1.0+np.log(x+1))):
    #neighborhood sizes for each node
    n_neighbors = np.squeeze(np.array(graph_local.sum(axis=1)))
    #map it into the AA weights as given by the function argument
    AA_weights = func(n_neighbors)
    
    #actually compute AA features
    AA_features = np.ndarray(len(data_local))
    for i,edge in enumerate(data_local):
        AA_features[i] = np.squeeze(np.array(graph_local.getrow(edge[0]).dot(graph_local.getrow(edge[1]).multiply(AA_weights).transpose())))    
    sys.stdout.flush()
    return AA_features

def get_common_neighbors_all(graph,idxs):
    n_entities = graph.shape[0]
    common_neighbors = np.zeros(n_entities*len(idxs))
    for i,idx in enumerate(idxs):
        common_neighbors[i*n_entities:(i+1)*n_entities] = graph.getrow(idx).dot(graph.transpose()).toarray()[0]
    return common_neighbors

def get_AA_all(graph,idxs,func=lambda x: 1/(1.0+np.log(x+1))):
    #neighborhood sizes for each node
    n_neighbors = np.squeeze(np.array(graph.sum(axis=1)))
    #map it into the AA weights as given by the function argument
    AA_weights = func(n_neighbors)

    n_entities = graph.shape[0]
    AA_features = np.zeros(n_entities*len(idxs))
    for i,idx in enumerate(idxs):
        AA_features[i*n_entities:(i+1)*n_entities] = np.squeeze(graph.dot(graph.getrow(idx).multiply(AA_weights).transpose()))

    return AA_features

''' testing '''

    # test_graph_data = np.array([[0,1],[1,2],[0,4],[2,4]])
    # test_graph = get_graph(test_graph_data,n_entities=5)

    # '''test individual ones'''
    # test_edges = np.array([[0,2],[1,4]])
    # get_common_neighbors(test_graph,test_edges)
    # get_AA(test_graph,test_edges)
    # x = featurize_samples(test_graph,test_edges)

    # '''test vector-producing ones'''
    # get_common_neighbors_all(test_graph,[0,4])
    # x = get_AA_all(test_graph,[0,4])

    # '''test featurizing-everything'''
    # x = featurize_all(test_graph,[0,4])

def gen_and_dump(dataset,N_TRAINING_SAMPLES,N_TEST_SAMPLES):
    x = model_observables()
    print "loading data"
    x.load_dataset(dataset=dataset,VALIDATION_HOLDOUT = 0.05)
    print "generating training data"
    x.gen_training_data(N_TRAINING_SAMPLES=N_TRAINING_SAMPLES,use_cache=False)
    print "training"
    x.train()
    print "generating results"
    x.gen_results(N_TEST_SAMPLES = N_TEST_SAMPLES)
    print "saving and exiting."
    x.dump_results()
    x.dump_model()
    return x

if __name__ == "__main__":
    # dataset = 'KG_233'
    # x = model_observables()
    # print "loading data"
    # x.load_dataset(dataset=dataset,VALIDATION_HOLDOUT = 0.05)
    # print "loading training data"
    # x.gen_training_data(N_TRAINING_SAMPLES=None,use_cache=True)
    # print "loading model"
    # x.load_model()
    # scores = x.score_neighbors(0)


    # gen_and_dump('Wordnet',100000,1000)
    # gen_and_dump('KG_233',100000,1000)
    # gen_and_dump('Slashdot',100000,1000)
    x = gen_and_dump('Flickr',100,10)
    #x = gen_and_dump('Blogcatalog',100000,1000)
#     #eval train set log loss, just for sanity
#     x = model_observables()
#     x.load_dataset(dataset='KG_233',VALIDATION_HOLDOUT=0.05)
#     print "loading model"
#     x.load_model()
#     print "loading training data"
#     x.gen_training_data(N_TRAINING_SAMPLES=100000,use_cache=True)

    print "computing loss"
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(x.features_train)
    print "train loss:", np.mean(-x.model.predict_log_proba(features_scaled)[np.arange(len(features_scaled)),x.labels_train])
#     pdb.set_trace()
#     x.load_results()
    indptr_test = x.graphs['test'].indptr
    test_connected = x.graphs['test'].indices[indptr_test[x.samples]]
    pos_loss = np.mean(-np.log(x.scores[np.arange(len(x.scores)),test_connected]))
    neg_loss = np.mean(-np.log(1-x.scores[np.arange(len(x.scores)),np.random.randint(x.n_entities,size=len(x.scores))]))
    print 'test pos and neg losses:',pos_loss,neg_loss
#     '''
#     (Pdb) neg_loss
# 0.2160848921876409
# (Pdb) pos_loss
# 0.38217087087719837

# :)
