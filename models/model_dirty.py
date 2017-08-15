import theano
import numpy
import numpy.random
import cPickle as pickle
from collections import OrderedDict
import os

from theano import tensor as T
from theano.tensor.nnet import sigmoid

theano.config.compute_test_value = 'off'
MODEL_DIR = '/Users/vmisra/data/kg_embed_data/'

class simple_linkpred(object):

    def __init__(self, dim, n_entities, idxs_1=None, idxs_2=None, labels=None, batch_size=None):
        self.__dict__.update(locals())
        del self.self


        #Start by defining the graph

        ###
        # <--dim-->
        # ^
        # |
        # n_entities
        # |
        # v
        self.emb = theano.shared( (numpy.random.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX))
        self.emb.tag.test_value = (numpy.random.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX)

        ###
        #2 scalar parameters
        self.a = theano.shared (numpy.asarray(1.0).astype(theano.config.floatX))
        self.b = theano.shared (numpy.asarray(0.0).astype(theano.config.floatX))

        self.params = [self.emb, self.a, self.b]
        self.names = ['embedding', 'hamming_scale', 'hamming_shift']

        ###
        # <--1-->
        # ^
        # |
        # minibatch_size
        # |
        # v     
        self.x1_idxs = T.ivector()
        self.x2_idxs = T.ivector()
        self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)
        
        test = self.emb[self.x1_idxs]

        ###
        # <--dim-->
        # ^
        # |
        # minibatch_size
        # |
        # v     
        #import pdb
        #pdb.set_trace()
        x1_emb = sigmoid(self.emb[self.x1_idxs].reshape((self.x1_idxs.shape[0],self.dim)))
        x2_emb = sigmoid(self.emb[self.x2_idxs].reshape((self.x2_idxs.shape[0],self.dim)))

        ### binary vector
        # ^
        # |
        # minibatch_size
        # |
        # v 
        self.y = T.ivector()
        self.y.tag.test_value = numpy.asarray([1,1],dtype=numpy.int32)

        ### float vector
        # ^
        # |
        # m
        # v inibatch_size
        # |

        self.p1 = sigmoid(self.a*T.mean(x1_emb*x2_emb + (1-x1_emb)*(1-x2_emb), axis=1)+self.b) #probability of a link, 0 to 1.'
        self.p0 = 1-self.p1
        
        ### float vector --- probability assigned to opposite of ground truth
        # ^
        # |
        # m
        # v inibatch_size
        # 
        ###self.loss = T.mean(T.switch(T.eq(self.y,0),p1,p0))
        self.loss = T.mean(-T.log(self.p1)*self.y - T.log(self.p0)*(1-self.y))


        ##training functions
        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1
        self.grads = T.grad(self.loss, self.params)
        updates = OrderedDict([(p, p-self.lr*g) for p,g in zip(self.params,self.grads)])

        self.train = theano.function( inputs = [self.x1_idxs, self.x2_idxs, self.y, self.lr],
                                      outputs = self.loss,
                                      updates = updates)

        if self.batch_size != None:
            self.shared_idx1 = theano.shared(self.idxs_1,borrow=True)
            self.shared_idx2 = theano.shared(self.idxs_2,borrow=True)
            self.shared_y = theano.shared(self.labels,borrow=True)

            n_batches = len(self.labels)/self.batch_size
            updates_minibatch = OrderedDict([(p, p-self.lr*g/n_batches) for p,g in zip(self.params,self.grads)])
            index = T.lscalar()
            self.minibatch_train = theano.function(inputs = [index, self.lr],
                                                outputs = self.loss,
                                                updates = updates_minibatch,
                                                givens = {self.x1_idxs : self.shared_idx1[index*self.batch_size:(index+1)*self.batch_size],
                                                          self.x2_idxs : self.shared_idx2[index*self.batch_size:(index+1)*self.batch_size],
                                                          self.y       : self.shared_y[index*self.batch_size:(index+1)*self.batch_size]})

    def save(self,subdir='scratch'):
        if not os.path.exists(MODEL_DIR+subdir):
            os.makedirs(MODEL_DIR+subdir)

        for p,name in zip(self.params,self.names):
            pickle.save(p,open(MODEL_DIR+subdir+name+'.pkl','w'))


def gen_fake_graph(n_entities, n_edges_approx):
    from numpy.random import randint
    edges1 = numpy.randint(0,n_entities,n_edges_approx)
    edges2 = numpy.oneslike(edges1)
    for i in range(len(edges2)):
        edges2[i] = randint(edges1[i]+1,n_entities)
    return [edges1,edges2]

def batch_train_test():
    edges = [[0,0,1,0,1,2],[1,2,2,3,3,3]]
    LP = simple_linkpred(1,4)

    def train_and_print():
        for _ in range(10000):
            curr_p = LP.train(edges[0],edges[1],[1,0,0,0,0,0],1)
        print curr_p

    train_and_print()
    print LP.p1.eval({LP.x1_idxs:edges[0],LP.x2_idxs:edges[1]})
    LP.p1.eval({LP.x1_idxs:edges[0],LP.x2_idxs:edges[1]})


#load KG graph
KG_path = '/ceph/vinith/kg/embedding/traindata_v233_minmentions2_styleuniform.pkl'
N_EDGES = 100

###from KG
KG_data = pickle.load(open(KG_path,'r'))

###fake data, old format
#KG_data = [numpy.transpose(numpy.asarray([[0,0,1,0,1,2],[1,2,2,3,3,3]],dtype=numpy.int32)),
#           numpy.asarray([1,0,0,0,0,0],dtype=numpy.int32)]

###turn to usable format, old format
#data_x = numpy.transpose(KG_data[0][:N_EDGES,:]).astype(numpy.int32)
#data_y = KG_data[1][:N_EDGES].flatten().astype(numpy.int32)

###new format
data_x1 = numpy.transpose(KG_data[0][:N_EDGES,:]).astype(numpy.int32)
#data_x0 = numpy.transpose(KG_data[1][:N_EDGES,:]).astype(numpy.int32)
data_y1 = KG_data[2][:N_EDGES].flatten().astype(numpy.int32)
#data_y0 = KG_data[3][:N_EDGES].flatten().astype(numpy.int32)

n_entities = max(data_x1.max(),data_x0.max())+1
dim = 20

import pdb
pdb.set_trace()

LP = simple_linkpred(dim=dim,
                         n_entities=n_entities,
                         idxs_1=data_x[0],
                         idxs_2=data_x[1],
                         labels=data_y,
                         batch_size=2)




