import theano
import math
import numpy
import numpy.random
import cPickle as pickle
from collections import OrderedDict

import os
import time

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from scipy.special import erfinv #for more precise expectation computation

theano.config.compute_test_value = 'off' #'warn' #
#MODEL_DIR = '/Users/vmisra/data/kg_embed_data/'

class simple_linkpred(object):
    '''Parameters:
    path_load: if model is to be loaded from a directory with saved parameters, path thereof.
    dim: width of embedding matrix.
    n_entities: height of embedding matrix
    batch_size: during optimization, how large a batch size to work with (None -> GD)
    n_samples: for testing the actual performance, how many times to sample from the embedding probability matrix.
    seed: for generating all the initial values, as well as the sampling of negative cases. None-> chosen randomly.
    parameterization: are we feeding the embeddings into a sigmoid ('SIGMOID') or using them directly ('DIRECT')?
    embedding_type: see list of options in build_graph. 'BIT','REAL', and variations.
    init_a/init_b: initial values for a and b.
    objective_samples: how many places in the normal-approximated distribution of <e_x,e_y> we should
                        sample to get the objective function. Sampling once means using the expectation,
                        and more than once means using erfinv to sample at uniformly spaced points in the inverse cdf
                        (normal approximation).

    '''
    def __init__(self, path_load = None,
                        dim=10, 
                        n_entities=100, 
                        batch_size=None, 
                        n_samples=None,
                        seed=None,
                        parameterization='SIGMOID',
                        embedding_type='BIT',
                        init_a = 1.0,
                        init_b = -.5,
                        objective_samples = None):
        
        self.__dict__.update(locals())
        del self.self
        
                #if load option is activated:
        if path_load == None:
            self.build_graph()
        else:
            self.load(path_load)

    def save(self,path_save):
        save_dict={}
        save_dict['params'] = [p.get_value() for p in self.params]
        hypers = [['dim',self.dim],['n_entities',self.n_entities],['batch_size',self.batch_size],['n_samples',self.n_samples]]
        for name,hyper in hypers:
            save_dict[name]=hyper
        pickle.dump(save_dict,open(path_save,'w'))
    

    def load(self,path_load):
        load_dict = pickle.load(open(path_load,'r'))
        self.dim = load_dict['dim']
        self.n_entities = load_dict['n_entities']
        self.batch_size = load_dict['batch_size']
        #legacy problem :/
        if 'validation_samples' in load_dict:
            self.n_samples = load_dict['validation_samples']
        else:
            self.n_samples = load_dict['n_samples']

        self.build_graph()

        for p,pval in self.params,load_dict['params']:
            p.set_value(pval)

    #loads model with snapshots in path_model_snapshots to a certain epoch/batch point using the training arguments in train_args
    def load_to_location(self,path_model_snapshots,train_args):
        raise NotImplementedError

    def build_graph(self):
        if self.seed==None:
            self.seed = numpy.random.randint(2**30)
        theano_rng = RandomStreams(self.seed)
        randstate = numpy.random.RandomState(self.seed)
        
        ##################
        ##Parameter setup
        ##################
        self.emb = theano.shared( (randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX))
        self.emb.tag.test_value = (randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX)

        self.a = theano.shared (numpy.asarray(self.init_a).astype(theano.config.floatX))
        self.b = theano.shared (numpy.asarray(self.init_b).astype(theano.config.floatX))

        self.params = [self.emb, self.a, self.b]

        if self.embedding_type=='REAL_TRAINED':
            self.coef_defaults = [2.0,-.5,-.5,-.5,-.5]
            self.coefs = [theano.shared(numpy.asarray(coef_default).astype(theano.config.floatX)) for coef_default in self.coef_defaults]
            self.params = self.params+self.coefs
        if self.embedding_type=='REAL_TRAINED_L1':
            self.coef_defaults = [2.0,-.5,-.5,0,0]
            self.coefs = [theano.shared(numpy.asarray(coef_default).astype(theano.config.floatX)) for coef_default in self.coef_defaults]
            self.params = self.params+self.coefs[:-2]

        ################
        ### Input setup!
        #################
        self.x1_idxs = T.ivector()
        self.x2_idxs = T.ivector()
        self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)
        
        #generate negative samples
        choice = theano_rng.binomial(size=self.x1_idxs.shape)
        alternative = theano_rng.random_integers(size=self.x1_idxs.shape,low=0,high=self.n_entities-1)
        self.x1_idxs_negative = T.switch(choice,self.x1_idxs,alternative)
        self.x2_idxs_negative = T.switch(choice,alternative,self.x2_idxs)

        ### Define graph from input to predictive loss
        def get_embed(index_tensor):
            #index_tensor: (samples)
            if self.parameterization=='SIGMOID':
                return sigmoid(self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim)))
            elif self.parameterization=='DIRECT':
                return self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim))
            
        self.x1_emb = get_embed(self.x1_idxs)
        self.x2_emb = get_embed(self.x2_idxs)
        self.x1neg_emb = get_embed(self.x1_idxs_negative)
        self.x2neg_emb = get_embed(self.x2_idxs_negative)

        def get_prob(embed_tensor1,embed_tensor2):
            #embed_tensorX: (n_batches,dim,*)
            if self.embedding_type=='BIT':
                return sigmoid(self.a*T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)+self.b) #returns (n_batches,_,*)
            if self.embedding_type=='BIT_INTERNALB':
                return sigmoid(self.a*(T.mean(2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0, axis=1)+self.b)) #returns (n_batches,_,*)
            if self.embedding_type=='BIT_AND':
                return sigmoid(self.a*T.mean(2.0*embed_tensor1*embed_tensor2, axis=1)+self.b) #returns (n_batches,_,*)
            elif self.embedding_type=='REAL':   
                return sigmoid(self.a*T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b) #returns (n_batches,_,*)
            elif self.embedding_type=='REAL_INTERNALB':   
                return sigmoid(self.a*(T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b)) #returns (n_batches,_,*)
            elif self.embedding_type=='REAL_SQRT':
                return sigmoid(self.a*(T.mean(1.0-(embed_tensor1-embed_tensor2)*(embed_tensor1-embed_tensor2), axis=1))**.5+self.b) #returns (n_batches,_,*)
            elif self.embedding_type=='REAL_L1':
                return sigmoid(self.a*(T.mean(1.0-T.abs_(embed_tensor1-embed_tensor2), axis=1))+self.b) #returns (n_batches,_,*)
            elif self.embedding_type=='REAL_TRAINED' or self.embedding_type=='REAL_TRAINED_L1':
                terms = [embed_tensor1*embed_tensor2, embed_tensor1, embed_tensor2,embed_tensor1**2,embed_tensor2**2]
                expr = sum([term*coef for term,coef in zip(terms,self.coefs)])
                return sigmoid(self.a*T.mean(expr, axis=1)+self.b)

        def get_prob_sampled(embed_tensor1,embed_tensor2,n_samples):
            randomizationA = theano_rng.uniform(size=(embed_tensor1.shape[0],embed_tensor1.shape[1],n_samples)) #(n_batches,dim,val)
            randomizationB = theano_rng.uniform(size=(embed_tensor2.shape[0],embed_tensor2.shape[1],n_samples)) #(n_batches,dim,val)
            bithash_1 = T.switch(T.lt(randomizationA,embed_tensor1.dimshuffle(0,1,'x')),1,0) #(val,dim)
            bithash_2 = T.switch(T.lt(randomizationB,embed_tensor2.dimshuffle(0,1,'x')),1,0) #(val,dim)
            return ([bithash_1,bithash_2],get_prob(bithash_1,bithash_2))

        def get_mean(embed_tensor1,embed_tensor2):
            return self.a*T.mean(2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2 +1.0+self.b, axis=1)

        def get_var(embed_tensor1, embed_tensor2):
            p = 2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0
            variances = p*(1-p)
            total_var = T.sum(variances,axis=1)*(self.a/T.shape(variances)[1])**2
            return total_var

        
        #build up list of sampling points and sampling weights, according to normal cdf approximation.
        #if objective_samples == None, stick to sub-optimal sampling scheme.
        def get_samples(embed_tensor1,embed_tensor2):
            if self.objective_samples == None:
                return[{'weight':1.0, 'value':get_prob(embed_tensor1,embed_tensor2)}]
            else:
                PhiInv = lambda z: 2**.5*erfinv(2*z-1)
                means = get_mean(embed_tensor1,embed_tensor2)
                variances = get_var(embed_tensor1,embed_tensor2)
                # print (variances**.5).tag.test_value
                spacing = 1.0/(self.objective_samples+1)
                xs = []
                for i in range(1,self.objective_samples+1):
                    sample = variances**.5*PhiInv(float(i)*spacing)+means
                    xs.append({'weight':1.0*spacing,
                               'value':sigmoid(sample)})
                xs.append({'weight':0.5*spacing,
                            'value':sigmoid(variances**.5*PhiInv(0.5*spacing)+means)})
                xs.append({'weight':0.5*spacing,
                            'value':sigmoid(variances**.5*PhiInv(1-0.5*spacing)+means)})
                return xs

        pos_losses = [-sample['weight']*T.mean(T.log(sample['value'])) for sample in get_samples(self.x1_emb,self.x2_emb)]
        neg_losses = [-sample['weight']*T.mean(T.log(1-sample['value'])) for sample in get_samples(self.x1neg_emb,self.x2neg_emb)]
        self.loss = sum(pos_losses+neg_losses)
        # for sample in get_samples(self.x1_emb,self.x2_emb):
        #     print "weight: ",sample['weight'], "value: ",sample['value'].tag.test_value
        # for x in pos_losses:
        #     print "pos loss test_value:",x.tag.test_value
        # for x in neg_losses:
        #     print "neg loss test_value:",x.tag.test_value
        #print "loss test value: ",self.loss.tag.test_value

        if self.n_samples != None:
            self.bithash_1s,self.bit_p1 = get_prob_sampled(self.x1_emb,self.x2_emb,self.n_samples)
            self.bithash_1s,self.bit_p2 = get_prob_sampled(self.x1neg_emb,self.x2neg_emb,self.n_samples)
            self.sampled_loss = T.mean(-T.log(self.bit_p1)-T.log(1-self.bit_p2))

    def get_training_fn(self,idxs,idxs_validate=None,training='SGD',debug='None'):
        ###################
        #initialize parameters
        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1
        if self.batch_size == None:
            batch_size = len(idxs)
        else:
            batch_size = self.batch_size
        n_batches = int(math.ceil(float(len(idxs))/batch_size))

        ###################
        #initialize data
        shared_idx1 = theano.shared(numpy.transpose(idxs)[0],borrow=True)
        shared_idx2 = theano.shared(numpy.transpose(idxs)[1],borrow=True)


        ###################
        #construct update graph
        self.grads = T.grad(self.loss, self.params)
        if training=='SGD':
            self.updates_minibatch = OrderedDict([(p, p-self.lr*g/n_batches) for p,g in zip(self.params,self.grads)])
        elif training=='ADAGRAD':
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
        outputs = [self.loss]
        outputnames = ['loss']
        if self.n_samples != None:
            outputs += [self.sampled_loss]
            outputnames += ['sampled loss']
        if debug=="NORMS":
            outputs = outputs+[T.sqrt(T.sum(self.grads[0]**2)),T.sqrt(T.sum(self.emb**2))]
            outputnames += ['grad norm','embedding norm']

        self.train = theano.function(inputs = [index, self.lr],
                                             outputs = outputs,
                                             updates = self.updates_minibatch,
                                             givens = {self.x1_idxs : shared_idx1[index*self.batch_size:(index+1)*self.batch_size],
                                                       self.x2_idxs : shared_idx2[index*self.batch_size:(index+1)*self.batch_size]})
        training_bundle.append(('train', self.train, outputnames))
        
        #next, validation
        if idxs_validate != None and len(idxs_validate)>0:
            self.shared_idx1_validate = theano.shared(numpy.transpose(idxs_validate)[0],borrow=True)
            self.shared_idx2_validate = theano.shared(numpy.transpose(idxs_validate)[1],borrow=True)
            valid_outputs = [self.loss]
            valid_outputnames = ['loss']
            if self.n_samples != None:
                valid_outputs += [self.sampled_loss]
                valid_outputnames += ['sampled loss']
            self.validate = theano.function(inputs = [],
                                                        outputs= outputs,
                                                        givens = {self.x1_idxs : self.shared_idx1_validate,
                                                                  self.x2_idxs : self.shared_idx2_validate})

            training_bundle.append(('validate', self.validate, valid_outputnames))
        
        #return training bundle of form [(optimization_function, names_for_each_reported_channel)]
        return training_bundle