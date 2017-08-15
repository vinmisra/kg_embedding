import theano
import numpy
import numpy.random
import cPickle as pickle
from collections import OrderedDict

import os
import time

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.compute_test_value = 'off'
MODEL_DIR = '/Users/vmisra/data/kg_embed_data/'

class simple_linkpred(object):

    def __init__(self, dim, n_entities, batch_size=None, validation_samples=2):

        self.__dict__.update(locals())
        del self.self
        
        theano_rng = RandomStreams(numpy.random.randint(2 ** 30))

        #Start by defining the graph

        ##Parameter setup
        self.emb = theano.shared( (numpy.random.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX))
        self.emb.tag.test_value = (numpy.random.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX)

        self.a = theano.shared (numpy.asarray(1.0).astype(theano.config.floatX))
        self.b = theano.shared (numpy.asarray(0.0).astype(theano.config.floatX))

        self.params = [self.emb, self.a, self.b]

        ### Input setup!
        self.x1_idxs = T.ivector()
        self.x2_idxs = T.ivector()
        self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)
        
        #generate negative samples
        choice = theano_rng.binomial(size=self.x1_idxs.shape)
        alternative = theano_rng.random_integers(size=self.x1_idxs.shape,low=0,high=n_entities-1)
        self.x1_idxs_negative = T.switch(choice,self.x1_idxs,alternative)
        self.x2_idxs_negative = T.switch(choice,alternative,self.x2_idxs)



        ### Define graph from input to predictive loss
        def get_embed(index_tensor):
            return sigmoid(self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim)))
            
        x1_emb = get_embed(self.x1_idxs)
        x2_emb = get_embed(self.x2_idxs)
        x1neg_emb = get_embed(self.x1_idxs_negative)
        x2neg_emb = get_embed(self.x2_idxs_negative)

        def get_prob1(embed_tensor1,embed_tensor2):
            return sigmoid(self.a*T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)+self.b) #probability of a link, 0 to 1.'

        self.loss = T.mean(-T.log(get_prob1(x1_emb,x2_emb)) - T.log(1-get_prob1(x1neg_emb,x2neg_emb)))

        ###Define graph from input to sampled/validated loss
        randomizationA = theano_rng.uniform(size=(self.validation_samples,self.dim))
        randomizationB = theano_rng.uniform(size=(self.validation_samples,self.dim))
        

    def get_training_fn(self,idxs):
        if self.batch_size == None:
            self.batch_size = len(idxs)

        self.shared_idx1 = theano.shared(numpy.transpose(idxs)[0],borrow=True)
        self.shared_idx2 = theano.shared(numpy.transpose(idxs)[1],borrow=True)

        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1

        self.grads = T.grad(self.loss, self.params)
        n_batches = len(idxs)/self.batch_size
        updates_minibatch = OrderedDict([(p, p-self.lr*g/n_batches) for p,g in zip(self.params,self.grads)])

        index = T.lscalar()
        self.train = theano.function(inputs = [index, self.lr],
                                     outputs = self.grads+[self.loss],
                                     updates = updates_minibatch,
                                     givens = {self.x1_idxs : self.shared_idx1[index*self.batch_size:(index+1)*self.batch_size],
                                               self.x2_idxs : self.shared_idx2[index*self.batch_size:(index+1)*self.batch_size]})
        
        return self.train

    def get_adagrad_fn(self,idxs):
        if self.batch_size == None:
            self.batch_size = len(idxs)

        self.shared_idx1 = theano.shared(numpy.transpose(idxs)[0],borrow=True)
        self.shared_idx2 = theano.shared(numpy.transpose(idxs)[1],borrow=True)

        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1

        self.grads = T.grad(self.loss, self.params)
        self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]

        n_batches = len(idxs)/self.batch_size
        updates_minibatch = OrderedDict([(p,T.switch(T.eq(g,0),p,p-(self.lr/n_batches)*g/((hg+g*g+0.000000000001)**.5))) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
        for hg,g in zip(self.history_grads,self.grads):
            updates_minibatch[hg]=hg + g*g

        index = T.lscalar()
        self.train_adagrad = theano.function(inputs = [index, self.lr],
                                     outputs = [self.loss],
                                     updates = updates_minibatch,
                                     givens = {self.x1_idxs : self.shared_idx1[index*self.batch_size:(index+1)*self.batch_size],
                                               self.x2_idxs : self.shared_idx2[index*self.batch_size:(index+1)*self.batch_size]})
        
        return self.train_adagrad
        

    def save(self,subdir='scratch'):
        if not os.path.exists(MODEL_DIR+subdir):
            os.makedirs(MODEL_DIR+subdir)

        pickle.save([p.get_value() for p in self.params],open(MODEL_DIR+subdir+'params.pkl','w'))

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

"""
#deprecated ---- testing now performed in KG testing notebook.
if __name__=='__main__':
    #graph params
    N_EDGES = 1000000
    dim = 20
    n_entities = 5000000

    ###real data
    data_x = KG_data[0][:N_EDGES,:].astype(numpy.int32)
    max_index = data_x.max()+1
    if n_entities < max_index:
        n_entities = max_index

    ###fake data
    #max_edge = 1900000
    #data_x = 



    LP = simple_linkpred(dim=dim,
                     n_entities=n_entities,
                     batch_size=None)

    train = LP.get_training_fn(data_x)

    start = time.time()
    train(0,1)
    end = time.time()
    print "batchsize:",N_EDGES
    print "dim:",dim
    print "n_entities:",n_entities
    print "time:",end - start
    print "time per 20M:",float(end-start)*20000000/float(N_EDGES)
"""
