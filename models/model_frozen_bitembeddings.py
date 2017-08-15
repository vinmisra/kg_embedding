import sys
import theano
import math
import numpy
import numpy as np
import numpy.random
import cPickle as pickle
from collections import OrderedDict

import os
import time

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from scipy.special import erfinv #for more precise expectation computation

import theano.sparse
from scipy.sparse import csr_matrix
theano.config.compute_test_value = 'off' #'warn' #

class frozen_embeddings(object):
    '''
    arguments:
    1. embeddings: can be either a numpy array of the embeddings, or a string path to where a model has been dumped by model_neighborhood.py
    2. seed: can be a number to sync randomness between multiple calls to this guy
    3. init_p_width: how much of a spread do you want the p-map from hamming to sigmoid to start with?
    4. quantization:    'SAMPLE': embeddings describe probabilities to be sampled from
                        'SIGMOID_SAMPLE', 'Sample: apply a sigmoid to embeddings and sample from output
                        'SIGMOID': just apply a sigmoid, leave it as a real-valued embedding.
                        'RHP': embeddings are real numbers, turn to bits using a random hyperplane LSH technique
                        'SH': embeddings are real numbers, turn to bits using the spectral hashing technique
                        'ITQ': embeddings are real numbers, turn to bits using iterative quantization.
    '''
    def __init__(self, 
                 parameters,
                 seed = None,
                 init_p_width = 2,
                 quantization = 'SAMPLE',
                 batch_size = None,
                 n_embed_bits = None,
                 neighborhood = None): 
        self.__dict__.update(locals())
        del self.self

        #first, load if need be
        if isinstance(parameters,basestring):
            print "Loading parameters"
            if not self.neighborhood:
                raw_embeddings, self.original_a, self.original_b = pickle.load(open(parameters,'r'))['params']
            else:
                raw_embeddings, self.original_a, self.original_b, self.neighborhood_transform, self.transform_scaling = pickle.load(open(parameters,'r'))['params']


        self.quantize_embeddings()
        self.n_entities = self.binary_embeddings.shape[0]
        self.dim = self.binary_embeddings.shape[1]

        self.embeddings = theano.shared(self.binary_embeddings,name='emb')
        self.build_graph_logloss()

    def quantize_embeddings(self):
        #first, copy over embeddings into local form
        raw_embeddings = self.raw_embeddings
        if not self.neighborhood:
            self.neighborhood_transform = numpy.eye(raw_embeddings.shape[1])
            self.transform_scaling = 1.0

        #second, generate actual embedding matrix


        if quantization == 'SIGMOID_SAMPLE':
            raw_embeddings = 1/(1+numpy.exp(-raw_embeddings))
            rand_thresh_mat = numpy.random.uniform(size=raw_embeddings.shape)
            return numpy.greater(raw_embeddings,rand_thresh_mat).astype(numpy.int32)
        elif quantization == 'SAMPLE' or quantization =='SAMPLE_RAW':
            rand_thresh_mat = numpy.random.uniform(size=raw_embeddings.shape)
            return numpy.greater(raw_embeddings,rand_thresh_mat).astype(numpy.int32)
        elif quantization == 'SH':
            if not n_embed_bits:
                n_embed_bits=raw_embeddings.shape[1]
            she=SHEncoder()
            pardic={'vals': raw_embeddings, 'nbits':n_embed_bits}
            she.build(pardic)
            return she.encode(raw_embeddings)
        elif quantization == 'RHP':
            if not n_embed_bits:
                n_embed_bits=raw_embeddings.shape[1]
            rhpe = RHPEncoder()
            rhpe.build(raw_embeddings.shape[1],n_embed_bits)
            return rhpe.encode(raw_embeddings)
        else:
            raise(NotImplementedError)

    def build_graph_logloss(self):
        #initialize for randomness
        if self.seed==None:
            self.seed = numpy.random.randint(2**30)
        theano_rng = RandomStreams(self.seed)
        self.randstate = numpy.random.RandomState(self.seed)

        #define parameters
        init_p_before_sigmoid = numpy.linspace(start=-self.init_p_width,stop=self.init_p_width, num=self.dim+1)
        self.p_before_sigmoid = theano.shared(init_p_before_sigmoid.astype(theano.config.floatX),name='p_before_sigmoid')
        self.params = [self.p_before_sigmoid]

        #define inputs
        self.x1_idxs = T.ivector()
        self.x2_idxs = T.ivector()
        self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)

        #define negative inputs
        choice = theano_rng.binomial(size=self.x1_idxs.shape)
        alternative = theano_rng.random_integers(size=self.x1_idxs.shape,low=0,high=self.n_entities-1)
        self.x1_idxs_negative = T.switch(choice,self.x1_idxs,alternative)
        self.x2_idxs_negative = T.switch(choice,alternative,self.x2_idxs)

        #define graph from inputs to probabilities and to log loss
        def get_embed(index_tensor):
            return self.embeddings[index_tensor].reshape((index_tensor.shape[0],self.dim))

        self.x1_emb = get_embed(self.x1_idxs)
        self.x2_emb = get_embed(self.x2_idxs)
        self.x1neg_emb = get_embed(self.x1_idxs_negative)
        self.x2neg_emb = get_embed(self.x2_idxs_negative)

        def get_prob(embed_tensor1,embed_tensor2):
            distances = T.sum(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)
            return sigmoid(self.p_before_sigmoid[distances])

        self.pos_probs = get_prob(self.x1_emb,self.x2_emb)
        self.neg_probs = get_prob(self.x1neg_emb,self.x2neg_emb)
        self.loss = -T.mean(T.log(self.pos_probs) + T.log(1.0-self.neg_probs))

    def get_training_fns_logloss(self,
                                 idxs_train,
                                 idxs_validate,
                                 algorithm='SGD'):
        ###################
        #initialize parameters
        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1
        if self.batch_size == None:
            batch_size = len(idxs_train)
        else:
            batch_size = self.batch_size
        n_batches = int(math.ceil(float(len(idxs_train))/batch_size)) 

        ###################
        #initialize data
        shared_idx1 = theano.shared(numpy.transpose(idxs_train)[0],borrow=True)
        shared_idx2 = theano.shared(numpy.transpose(idxs_train)[1],borrow=True)
        shared_idx1_val = theano.shared(numpy.transpose(idxs_validate)[0],borrow=True)
        shared_idx2_val = theano.shared(numpy.transpose(idxs_validate)[1],borrow=True)

        ###################
        #construct update graph
        self.grads = T.grad(self.loss, self.params)
        if algorithm=='SGD':
            self.updates_minibatch = OrderedDict([(p, p-self.lr*g/n_batches) for p,g in zip(self.params,self.grads)])
        elif algorithm=='ADAGRAD':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.switch(T.eq(g,0),p,p-(self.lr/n_batches)*g/((hg+g*g+0.000000000001)**.5))) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=hg + g*g
        else:
            print 'unrecognized training algorithm'
            return -1

        #construct optimization functions
        training_bundle = []

        #first, training
        index = T.lscalar()
        train_outputs = [self.loss]
        train_outputnames = ['loss']
        self.train = theano.function(inputs = [index, self.lr],
                                             outputs = train_outputs,
                                             updates = self.updates_minibatch,
                                             givens = {self.x1_idxs : shared_idx1[index*self.batch_size:(index+1)*self.batch_size],
                                                       self.x2_idxs : shared_idx2[index*self.batch_size:(index+1)*self.batch_size]})
        training_bundle.append(('train', self.train, train_outputnames))

        #then, validation
        valid_outputs = [self.loss]
        valid_outputnames = ['loss']
        self.validate = theano.function(inputs = [],
                                        outputs= valid_outputs,
                                        givens = {self.x1_idxs : shared_idx1_val,
                                                  self.x2_idxs : shared_idx2_val})
        training_bundle.append(('validate', self.validate, valid_outputnames))

        return training_bundle

    def test_logloss(self, idxs_test1, idxs_test2):
        raise(NotImplementedError)

    #computes "discounted cumulative"
    def test_DCG(self, idxs_test1, idxs_test2):
        idxs_embedded1 = self.binary_embeddings[idxs_test1]
        hamming_similarity = idxs_embedded1.dot(self.binary_embeddings.T) + (1-idxs_embedded1).dot(1-self.binary_embeddings.T)
        sorted_by_similarity = np.argsort(-hamming_similarity,axis=-1)
        idxs_test2_rp = np.repeat(idxs_test2.reshape(idxs_test2.shape[0],1),repeats=self.n_entities,axis=1)
        ranks = np.where(sorted_by_similarity == idxs_test2_rp)[1]
        scores = 1/numpy.log(numpy.clip(ranks+1,a_min=2,a_max=1000))
        return sum(scores)/len(scores)

class Encoder(object):
    """
    Encoder maps original data to hash codes.
    """
    def __init__(self):
        self.ERR_INSTAN = "Instance of `Encoder` is not allowed!"
        self.ERR_UNIMPL = "Unimplemented method!"
        pass

    def __del__(self):
        pass

    def build(self, vals=None, labels=None):
        """
        Build the encoder based on given training data.
        """
        raise Exception(self.ERR_INSTAN)

    def load(self, path):
        """
        Load encoder information from file.
        """
        with open(path, 'rb') as pklf:
            self.ecdat = pickle.load(pklf)

    def save(self, path):
        """
        Save the encoder information.
        """
        with open(path, 'wb') as pklf:
            pickle.dump(self.ecdat, pklf, protocol=2)

    def encode(self, vals):
        """
        Map `vals` to hash codes.
        """
        raise Exception(self.ERR_INSTAN)

''' implementation of random hyperplane LSH '''
class RHPEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)

    def build(self, dim, nbits):
        self.hash_table = numpy.random.normal(size=(dim, nbits))

    def encode(self, vals):
        return numpy.greater(vals.dot(self.hash_table),0).astype(numpy.int32)


''' implementation of Semantic hashing'''

def eigs(X, npca):
    l, pc = np.linalg.eig(X)
    idx = l.argsort()[::-1][:npca]
    return pc[:, idx], l[idx]

class SHEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)

    def __del__(self):
        pass

    def build(self, pardic=None):
        # training data
        X = pardic['vals']
        # the number of subquantizers
        nbits = pardic['nbits']
        # the number of items in one block
        blksize = pardic.get('blksize', 16384)

        [Nsamples, Ndim] = X.shape

        # algo:
        # 1) PCA
        npca = min(nbits, Ndim)
        pc, l = eigs(numpy.cov(X.T), npca)
        X = X.dot(pc)   # no need to remove the mean

        # 2) fit uniform distribution
        eps = numpy.finfo(float).eps
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        # 3) enumerate eigenfunctions
        R = mx - mn
        maxMode = numpy.ceil((nbits+1) * R / R.max())
        nModes = maxMode.sum() - maxMode.size + 1
        modes = numpy.ones((nModes, npca))
        m = 0
        for i in xrange(npca):
            modes[m+1:m+maxMode[i], i] = numpy.arange(1, maxMode[i]) + 1
            m = m + maxMode[i] - 1
        modes = modes - 1
        omega0 = numpy.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(nModes, 0)
        eigVal = -(omegas ** 2).sum(1)
        ii = (-eigVal).argsort()
        modes = modes[ii[1:nbits+1], :]

        """
        Initializing encoder data
        """
        ecdat = dict()
        ecdat['nbits'] = nbits
        ecdat['pc'] = pc
        ecdat['mn'] = mn
        ecdat['mx'] = mx
        ecdat['modes'] = modes
        ecdat['blksize'] = blksize
        self.ecdat = ecdat

    def encode(self, vals):
        X = vals
        if X.ndim == 1:
            X = X.reshape((1, -1))

        Nsamples, Ndim = X.shape
        nbits = self.ecdat['nbits']
        mn = self.ecdat['mn']
        mx = self.ecdat['mx']
        pc = self.ecdat['pc']
        modes = self.ecdat['modes']

        X = X.dot(pc)
        X = X - mn.reshape((1, -1))
        omega0 = 0.5 / (mx - mn)
        omegas = modes * omega0.reshape((1, -1))

        U = np.zeros((Nsamples, nbits))
        for i in range(nbits):
            omegai = omegas[i, :]
            ys = X * omegai + 0.25
            ys -= np.floor(ys)
            yi = np.sum(ys < 0.5, 1)
            U[:, i] = yi

        b = np.require(U % 2 == 0, dtype=np.int)
        B = b#self.compactbit(b)
        return B
