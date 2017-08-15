PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
VAL_FRAC = 0.05
SEED = 2052016

import sys, os, pdb
import model_testing
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import scipy
from sklearn.decomposition import TruncatedSVD
import cPickle as pickle

class model_fiedler(object):
    #must be given either a dataset_name or a graph
    def __init__(self, dim, laplacian_type,dataset_name=None,graph=None):
        self.dim = dim
        self.laplacian_type = laplacian_type
        self.seed= SEED
        
        self.dataset_name = dataset_name
        self.graph = graph

        if self.graph==None:
            self.load_dataset(self.dataset_name)

        self.laplacian = compute_laplacian(self.graph,self.laplacian_type)
        self.raw_embeddings = get_randSVD_emb(self.laplacian,self.dim)

    def get_embeddings(self):
        return self.raw_embeddings

    def dump_embeddings(self):
        filename = self.dataset_name+'__'+str(self.dim)+'__'+self.laplacian_type
        path_dump = os.path.join(BASE_DIR,'experiments','Fiedler',filename)
        pickle.dump(self.raw_embeddings,open(path_dump,'w'))

    '''
    loads both train and test datasets given seed parameter
    '''
    def load_dataset(self,dataset_name):
        #hardcoded paths to datasets
        paths = {}
        paths['KG'] = os.path.join(BASE_DIR,'data/traindata_db233_minmentions10_minentity3.pkl')
        paths['Wordnet'] = os.path.join(BASE_DIR,'data/noun_relations.pkl')
        paths['Slashdot'] = os.path.join(BASE_DIR,'data/slashdot.pkl')

        if dataset_name.startswith('KG'):
            data,entity_to_idx = pickle.load(open(paths['KG'],'r'))
            
        elif dataset_name=='Wordnet':
            data,idx_to_entity,entity_to_idx = pickle.load(open(paths[dataset_name],'r'))

        elif dataset_name=='Slashdot':
            raise NotImplementedError

        randstate = np.random.RandomState(self.seed)
        data = randstate.permutation(data)
        
        data_train,data_test = train_test_split(data,test_size=VAL_FRAC,random_state=self.seed)
        self.dataset= data_train

        ##Now load the graphs for each of the three subsets (train, test, all)
        n_entities = max(data.flatten())+1
        self.graph = get_graph(data_train,n_entities)

def compute_laplacian(graph,laplacian_type):
    D = dia_matrix((graph.sum(axis=0),[0]),shape=graph.shape)
    Dinv = dia_matrix((np.power(graph.sum(axis=0),-1),[0]),shape=graph.shape)
    Dinvrt = dia_matrix((np.power(graph.sum(axis=0),-.5),[0]),shape=graph.shape)

    if laplacian_type == 'unnormalized':
        return graph-D
    elif laplacian_type == 'symmetric':
        return Dinvrt.dot((graph-D).dot(Dinvrt))
    elif laplacian_type == 'random_walk':
        return Dinv.dot(graph-D)
    else:
        raise NotImplementedError


def get_randSVD_emb(mat,dim):
    svd = TruncatedSVD(n_components = dim)
    mat_xformed = svd.fit_transform(mat)
    return mat_xformed

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

'''test code '''
if __name__ == '__main__':
    x = model_fiedler(dataset_name='Wordnet',dim=25,laplacian_type='unnormalized')
    pdb.set_trace()
# def get_exactSVD_emb(mat,dim):
#     start=time.time()
#     svd = eigsh(mat,dim)
#     mat_xformed = svd.fit_transform(mat)
#     print time.time()-start
#     return mat_xformed