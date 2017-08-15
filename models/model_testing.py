import sys,pdb
import theano
import math
import numpy
import numpy as np
import numpy.random
import cPickle as pickle
from collections import OrderedDict

import os
import time

import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from scipy.special import erfinv #for more precise expectation computation
from scipy.linalg import sqrtm, inv
from numpy.linalg import svd

import theano.sparse
from scipy.sparse import csr_matrix
theano.config.compute_test_value = 'off' #'warn' #

class model_tester(object):
    '''
    arguments:
    1. embeddings: can be either a numpy array of the embeddings, or a string path to where a model has been dumped by model_neighborhood.py
    2. seed: can be a number to sync randomness between multiple calls to this guy
    3. init_p_width: how much of a spread do you want the p-map from hamming to sigmoid to start with?
    4. quantization:    'Sample': sample to get bit embeddings
                        'Raw': just leave it as is
                        'RHP': embeddings are real numbers, turn to bits using a random hyperplane LSH technique
                        'SH': embeddings are real numbers, turn to bits using the spectral hashing technique
                        'ITQ': embeddings are real numbers, turn to bits using iterative quantization.
    5. sigmoid: Apply a sigmoid to the raw embedding parameters before sampling/quantizing? (default is True)
    '''
    def __init__(self, 
                 parameters,
                 original_a = None,
                 original_b = None,
                 seed = None,
                 init_p_width = 2,
                 quantization = 'Sample',
                 mapping = 'Learned', #Learned, Hamming, L2
                 apply_sigmoid = True,
                 n_embed_bits = None,
                 graph_train = None): 
        self.__dict__.update(locals())
        del self.self

        #first, load if need be
        if isinstance(parameters,basestring):
            print "Loading parameters"
            parameters = pickle.load(open(parameters,'r'))['params']
        
        self.raw_embeddings,self.original_a,self.original_b = parameters[:3]

        if len(parameters) >3:
            self.neighborhood_transform, self.transform_scaling = parameters[3:]
            self.raw_embeddings = neighborhoodize_embeddings(self.raw_embeddings,self.neighborhood_transform, self.transform_scaling, self.graph_train)


        self.binary_embeddings = self.quantize_embeddings(self.raw_embeddings,quantization,n_embed_bits,apply_sigmoid=apply_sigmoid).astype(np.float32)
        self.n_entities = self.binary_embeddings.shape[0]
        self.dim = self.binary_embeddings.shape[1]

        self.embeddings = theano.shared(self.binary_embeddings,name='emb')
        self.build_graph_logloss()

    def isInverted(self):
        return (self.original_a < 0)

    def quantize_embeddings(self,raw_embeddings,quantization,n_embed_bits=None,apply_sigmoid=True):
        if apply_sigmoid:
            raw_embeddings = 1/(1+numpy.exp(-raw_embeddings))
        if quantization == 'Sample':
            randstate = numpy.random.RandomState(self.seed)
            rand_thresh_mat = randstate.uniform(size=raw_embeddings.shape)
            return numpy.greater(raw_embeddings,rand_thresh_mat).astype(numpy.int32)
        if quantization == 'Sample_deterministic':
            thresh_mat = 0.5*np.ones(raw_embeddings.shape)
            return numpy.greater(raw_embeddings,thresh_mat).astype(numpy.int32)
        elif quantization == 'Raw':
            return raw_embeddings
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
        elif quantization == 'ITQ':
            if not n_embed_bits:
                n_embed_bits=raw_embeddings.shape[1]
            itqe = ITQEncoder()
            return itqe.encode(raw_embeddings,dim=n_embed_bits)
        else:
            raise NotImplementedError


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

        #####TO MAKE MORE COMPARABLE WITH MODEL_NEIGHBORHOOD: replacing theano randomness with external randomness
        self.x1_idxs = theano.shared(numpy.asarray([0,1],dtype=numpy.int32))
        self.x2_idxs = theano.shared(numpy.asarray([2,3],dtype=numpy.int32))
        
        self.x1_idxs_negative = theano.shared(numpy.asarray([4,5],dtype=numpy.int32))
        self.x2_idxs_negative = theano.shared(numpy.asarray([3,5],dtype=numpy.int32))

        ###THE ORIGINAL 
        # #define inputs
        # self.x1_idxs = T.ivector()
        # self.x2_idxs = T.ivector()
        # self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        # self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)

        ##define negative inputs
        # choice = theano_rng.binomial(size=self.x1_idxs.shape)
        # alternative = theano_rng.random_integers(size=self.x1_idxs.shape,low=0,high=self.n_entities-1)
        # self.x1_idxs_negative = T.switch(choice,self.x1_idxs,alternative)
        # self.x2_idxs_negative = T.switch(choice,alternative,self.x2_idxs)


        #define graph from inputs to probabilities and to log loss
        def get_embed(index_tensor):
            return self.embeddings[index_tensor].reshape((index_tensor.shape[0],self.dim))

        self.x1_emb = get_embed(self.x1_idxs)
        self.x2_emb = get_embed(self.x2_idxs)
        self.x1neg_emb = get_embed(self.x1_idxs_negative)
        self.x2neg_emb = get_embed(self.x2_idxs_negative)


        

        def get_prob_learned(embed_tensor1,embed_tensor2):
            distances = T.sum(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1).astype('int32')
            return sigmoid(self.p_before_sigmoid[distances])
        def get_prob_hamming(embed_tensor1,embed_tensor2):
            distances = T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)
            return sigmoid(self.original_a*(distances+self.original_b))
        def get_prob_L2(embed_tensor1,embed_tensor2):
            distances = T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)
            return sigmoid(self.original_a*(distances+self.original_b))

        #####splits in two directions: adapted, and original (using a and b as loaded)
        if self.mapping == 'Hamming':
            get_prob = get_prob_hamming
        elif self.mapping == 'L2':
            get_prob = get_prob_L2
        elif self.mapping == 'Learned':
            get_prob = get_prob_learned

        self.pos_probs = get_prob(self.x1_emb,self.x2_emb)
        self.neg_probs = get_prob(self.x1neg_emb,self.x2neg_emb)
        self.loss = -T.mean(T.log(self.pos_probs) + T.log(1.0-self.neg_probs))


    ###performs training/adaptation and returns loss after ___ epochs.
    def train_logloss(self, idxs_train, lr=20,epochs = 20, algorithm='ADAGRAD'):
        ###################
        #initialize parameters
        batch_size = len(idxs_train)
        n_batches = int(math.ceil(float(len(idxs_train))/batch_size)) 

        ###################
        #initialize data
        shared_idx1 = theano.shared(numpy.transpose(idxs_train.astype(np.int32))[0],borrow=True)
        shared_idx2 = theano.shared(numpy.transpose(idxs_train.astype(np.int32))[1],borrow=True)

        ###################
        #construct update graph
        self.grads = T.grad(self.loss, self.params)
        if algorithm=='SGD':
            self.updates_minibatch = OrderedDict([(p, p-lr*g/n_batches) for p,g in zip(self.params,self.grads)])
        elif algorithm=='ADAGRAD':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.switch(T.eq(g,0),p,p-(lr/n_batches)*g/((hg+g*g+0.000000000001)**.5))) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=hg + g*g
        else:
            print 'unrecognized training algorithm'
            return -1


        #training function
        index = T.lscalar()
        train_outputs = self.loss
        self.train = theano.function(inputs = [index],
                                             outputs = train_outputs,
                                             updates = self.updates_minibatch,
                                             givens = {self.x1_idxs : shared_idx1[index*batch_size:(index+1)*batch_size],
                                                       self.x2_idxs : shared_idx2[index*batch_size:(index+1)*batch_size]})

        #################################
        # Perform training loop
        #################################
        for epoch in xrange(epochs):
            for batch in xrange(n_batches):
                loss = self.train(batch)
                print "latest loss: ", loss

        return loss

    '''
    tests logloss on given set
    original_or_adapted:
        'original': use the a and b params stored in the model
        'adapted': use the distance lookup params learned via train_logloss
    '''

    def test_logloss(self, idxs_test):
        #first, obtain values for input shared variables (x1,x2,x1neg,x2neg)
        def get_neg_samples_valid():
            idxs_neg_idx = np.random.permutation(range(len(idxs_test)))
            idxs_neg = idxs_test.copy()
            idxs_neg[idxs_neg_idx[:len(idxs_neg)/2],0] = np.random.randint(self.n_entities,size=len(idxs_neg)/2)
            idxs_neg[idxs_neg_idx[len(idxs_neg)/2:],1] = np.random.randint(self.n_entities,size=len(idxs_neg) - (len(idxs_neg)/2))
            idxs_neg = np.transpose(idxs_neg)
            return (idxs_neg[0],idxs_neg[1])

        x1_idxs_validate = np.transpose(idxs_test)[0].astype(np.int32)
        x2_idxs_validate = np.transpose(idxs_test)[1].astype(np.int32)
        x1_idxs_negative_validate,x2_idxs_negative_validate = get_neg_samples_valid()

        #set values for input shared variables
        self.x1_idxs.set_value(x1_idxs_validate)
        self.x2_idxs.set_value(x2_idxs_validate)
        self.x1_idxs_negative.set_value(x1_idxs_negative_validate.astype(np.int32))
        self.x2_idxs_negative.set_value(x2_idxs_negative_validate.astype(np.int32))

        return self.loss.eval()

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
            modes[int(m+1):int(m+maxMode[i]), i] = numpy.arange(1, int(maxMode[i])) + 1
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

''' implementation of iterative quantization LSH'''
''' first, some utilities'''
from scipy.linalg import sqrtm, inv

def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))

def get_random_orthogonal(dim):
    W = numpy.random.normal(size=(dim,dim))
    return sym(W)

'''next the actual quantizer'''
class ITQEncoder(Encoder):
    def __init__(self):
        Encoder.__init__(self)

    def encode(self, emb_mat, dim=None, n_iter = 20):
        return self.ITQ(n_iter=n_iter, data_mat=emb_mat, dim=dim)
    
    def initialize_ITQ(self, emb_mat,dim):    
        #zero mean
        emb_mat = emb_mat - np.mean(emb_mat,axis=0)
        
        #covariance + SVD
        covariance = np.dot(emb_mat.T,emb_mat)
        u,cov_xform,v = svd(covariance)
        
        #initialize R matrix
        R_init = get_random_orthogonal(dim)
        
        return (u,emb_mat.dot(v[:,:dim]),R_init)

    def iterate_ITQ(self, Rmat,XinPCA):
        #E step
        Bnew = XinPCA.dot(Rmat)>0
        
        #M step
        u,s,v = svd(Bnew.T.dot(XinPCA))
        Rnew = v.T.dot(u)
        
        return (Bnew,Rnew)

    def ITQ(self, n_iter,data_mat,dim=None):
        if dim == None:
            dim = data_mat.shape[1]
        u,data_xformed,R = self.initialize_ITQ(data_mat,dim)
        for _ in range(n_iter):
            B,R = self.iterate_ITQ(R,data_xformed)
        
        return B

def neighborhoodize_embeddings(raw_embeddings,neighborhood_transform,transform_scaling,graph_train):
    connection_embedding = graph_train.dot(raw_embeddings)
    print 'tscaling: ',transform_scaling
    xformed_embedding = transform_scaling*connection_embedding.dot(neighborhood_transform) + raw_embeddings
    return xformed_embedding
