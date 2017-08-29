PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
import os, sys, time
if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)

import theano, pdb
import math
import numpy
import numpy as np
import numpy.random
import cPickle as pickle
from collections import OrderedDict

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from scipy.special import erfinv #for more precise expectation computation

import theano.sparse
from scipy.sparse import csr_matrix

import utils.utils as utils
import utils.neg_sampling as neg_sampling
theano.config.compute_test_value = 'off' #'warn' #


class NCE_linkpred(object):
 
    '''
    For description of parameters in hypers, see hypers_NCE.py
    data_train and data_test are np.arrays of size (samples,2)
    data_train and data_test are assumed to be directed (not yet symmetrized).
    '''
    def __init__(self, hypers, data_train, data_test):
        self.hypers= hypers

        #setup data for training
        self.data_train = data_train
        self.data_test = data_test
        if self.hypers['DATA_SYM_TRAINTEST']:
            self.data_train = np.concatenate([data_train,data_train[:,::-1]],axis=0)
            self.data_test = np.concatenate([data_test,data_test[:,::-1]],axis=0)

        #setup data for NCE --- must always be symmetric
        self.graph_train_NCE = utils.get_graph(data_train) #graph must always be symmetric.
        if self.hypers['DATA_SYM_NCE']:
            self.data_train_NCE = np.concatenate([data_train,data_train[:,::-1]],axis=0)
        else:
            self.data_train_NCE  = data_train

        #shorthand for useful parameters
        self.dim = self.hypers['DIM']
        self.seed = self.hypers['SEED']
        self.n_entities = self.hypers['N_ENTITIES']
        self.lr = float(self.hypers['LEARNING_RATE'])
        self.batch_size_train = self.hypers['BATCH_SIZE']
        self.n_edges_train = len(self.data_train)
        self.n_batches_train = int(math.ceil(float(self.n_edges_train)/self.batch_size_train)) 

        #setup randomness
        if self.seed == None:
            self.seed = numpy.random.randint(2**30)
        self.randstate = numpy.random.RandomState(self.seed)

        #setup graph for evaluating loss/etc.
        self.build_graph()

        #setup negative sampler
        if self.hypers['NCE_DIST'] == 'UNIGRAM':
            self.neg_sampler = neg_sampling.UnigramSampler(data=self.data_train_NCE,power=self.hypers['UNIGRAM_POWER'],laplace=self.hypers['UNIGRAM_LAPLACE'],randstate=self.randstate)
        elif self.hypers['NCE_DIST'] == 'GRAPH_MARKOV':
            unigram_sampler = neg_sampling.UnigramSampler(data=self.data_train_NCE,power=self.hypers['UNIGRAM_POWER'],laplace=self.hypers['UNIGRAM_LAPLACE'],randstate=self.randstate)
            graph_sampler =  neg_sampling.GraphSampler(graph=self.graph_train_NCE, dist_power=self.hypers['NCE_GRAPH_DIST_POWER'], self_path_weight=self.hypers['NCE_GRAPH_SELF_WEIGHT'], balance_1st_vs_2nd_degree=self.hypers['NCE_GRAPH_BALANCE_1_2'],randstate=self.randstate)
            self.neg_sampler = neg_sampling.MixtureSampler(unigram_sampler=unigram_sampler,graph_sampler=graph_sampler,prob_unigram=self.hypers['NCE_MIX_PROB_UNIGRAM'])
        else:
            raise NotImplementedError

    def save(self,path_save):
        save_dict={}
        save_dict['params'] = [p.get_value() for p in self.params]
        for name,hyper in self.hypers.items():
            save_dict[name]=hyper

        pickle.dump(save_dict,open(path_save,'w'))

    def build_graph(self):
        theano_rng = RandomStreams(self.seed)
        
        ##################
        ##Parameter setup
        ##################
        self.emb = theano.shared((self.randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX),name="emb")
        # self.emb.tag.test_value = (self.randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX)

        self.a = theano.shared (numpy.asarray(self.hypers['INIT_A']).astype(theano.config.floatX),name='a')


        if self.hypers['NCE_B_CORRECTION']:
            self.hypers['INIT_B'] -= np.log(self.n_entities)
        self.b = theano.shared (numpy.asarray(self.hypers['INIT_B']).astype(theano.config.floatX),name='b')

        self.params = [self.emb, self.a, self.b]

        ################
        ### Input setup!
        #################
        self.x1_idxs = theano.shared(numpy.asarray([0,1],dtype=numpy.int32))
        self.x2_idxs = theano.shared(numpy.asarray([2,3],dtype=numpy.int32))
        
        self.x1_idxs_negative = theano.shared(numpy.asarray([0,5],dtype=numpy.int32))
        self.x2_idxs_negative = theano.shared(numpy.asarray([2,3],dtype=numpy.int32))

        self.NCEprobs = theano.shared(numpy.asarray([.2,.2],dtype=theano.config.floatX))
        self.NCEprobs_neg = theano.shared(numpy.asarray([.2,.2],dtype=theano.config.floatX))

        ####################
        ### Define embedding
        ####################
        '''
        returns embedding
        '''
        def get_embed(index_tensor):
            return sigmoid(self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim)))

        '''
        returns actual quantized/sampled version of embedding
        '''
        def get_embed_sampled(embed_tensor,sample=False):
            if not sample:
                randomization = 0.5
            else:
                randomization = theano_rng.uniform(size=embed_tensor.shape)
            return T.switch(T.lt(randomization,embed_tensor),1.0,0.0) #(val,dim)

        self.x1_emb = get_embed(self.x1_idxs)
        self.x2_emb = get_embed(self.x2_idxs)
        self.x1neg_emb = get_embed(self.x1_idxs_negative)
        self.x2neg_emb = get_embed(self.x2_idxs_negative)
        

        '''
        Conversions from pairs of embeddings to probabilities.
        '''

        #raw mapping from embedding pairs to logit
        def get_prob_logit(embed_tensor1,embed_tensor2,etype=self.hypers['EMBEDDING_TYPE']):
            if etype=='BIT':
                return self.a*T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)+self.b
            elif etype=='BIT_INTERNALB':
                return self.a*(T.mean(2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0, axis=1)+self.b) 
            elif etype=='BIT_AND':
                return self.a*T.mean(2.0*embed_tensor1*embed_tensor2, axis=1)+self.b 
            elif etype=='REAL':   
                return self.a*T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b 
            elif etype=='REAL_INTERNALB':   
                return self.a*(T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b) 
            elif etype=='REAL_SQRT':
                return self.a*(T.mean(1.0-(embed_tensor1-embed_tensor2)*(embed_tensor1-embed_tensor2), axis=1))**.5+self.b 
            elif etype=='REAL_L1':
                return self.a*(T.mean(1.0-T.abs_(embed_tensor1-embed_tensor2), axis=1))+self.b 

        #adds sigmoid and subtracts out nce_probs if they are provided.
        def get_prob(embed_tensor1, embed_tensor2, nce_probs=None):
            if type(nce_probs)==type(None):
                return sigmoid(get_prob_logit(embed_tensor1,embed_tensor2))  
            else:
                return sigmoid(get_prob_logit(embed_tensor1,embed_tensor2)-T.log(nce_probs))

        #functions as the above, but performs sampling from the bernoulli model
        #sample indicates whether to deterministically (false) or randomly (true) sample.
        def get_prob_sampled(embed_tensor1,embed_tensor2,sample=False,nce_probs=None):
            bithash_1 = get_embed_sampled(embed_tensor1,sample) #T.switch(T.lt(randomizationA,embed_tensor1.dimshuffle(0,1,'x')),1,0) #(val,dim)
            bithash_2 = get_embed_sampled(embed_tensor2,sample)#T.switch(T.lt(randomizationB,embed_tensor2.dimshuffle(0,1,'x')),1,0) #(val,dim)

            if nce_probs != None:
                return ([bithash_1,bithash_2],get_prob(bithash_1,bithash_2,nce_probs))
            else:
                return ([bithash_1,bithash_2],get_prob(bithash_1,bithash_2))


        def get_mean_logit(embed_tensor1,embed_tensor2):
            return get_prob_logit(embed_tensor1,embed_tensor2)

        def get_var_logit(embed_tensor1, embed_tensor2):
            p = 2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0
            variances = p*(1-p)
            total_var = T.sum(variances,axis=1)*(self.a/T.shape(variances)[1])**2
            return total_var

        #build up list of sampling points and sampling weights, according to normal cdf approximation.
        #if objective_samples == None, stick to sub-optimal sampling scheme.
        #if objective_samples < 0, sample (-objective_samples) from a normal distribution with the approximation (monte carlo)
        def get_samples_logit(embed_tensor1,embed_tensor2):
            if self.hypers['OBJECTIVE_SAMPLES'] == None:
                return[{'weight':1.0, 'value':get_prob_logit(embed_tensor1,embed_tensor2)}]
            elif self.hypers['OBJECTIVE_SAMPLES'] > 0:
                PhiInv = lambda z: 2**.5*erfinv(2*z-1)
                means = get_mean_logit(embed_tensor1,embed_tensor2)
                variances = get_var_logit(embed_tensor1,embed_tensor2) + 0.00000001

                spacing = 1.0/(self.hypers['OBJECTIVE_SAMPLES'] )
                xs = []
                for i in range(self.hypers['OBJECTIVE_SAMPLES']):
                    raw_sample = variances**.5*PhiInv(float(i+.5)*spacing)+means
                    xs.append({'weight':1.0*spacing,
                               'value':raw_sample})
                return xs
            elif self.hypers['OBJECTIVE_SAMPLES'] < 0:
                normals = theano_rng.normal(size=(-self.hypers['OBJECTIVE_SAMPLES'],embed_tensor1.shape[0]))
                means = get_mean_logit(embed_tensor1,embed_tensor2)
                variances = get_var_logit(embed_tensor1,embed_tensor2) + 0.00000001

                samples = normals*(variances**.5)+means 
                weighting = 1.0/(-self.hypers['OBJECTIVE_SAMPLES'])

                xs=[]
                for i in range(-self.hypers['OBJECTIVE_SAMPLES']):
                    xs.append({'weight':weighting,
                               'value': samples[i,:]})
                return xs

        def get_samples(embed_tensor1,embed_tensor2,nce_probs=None):
            samples = get_samples_logit(embed_tensor1,embed_tensor2)
            if nce_probs != None:
                for sample in samples:
                    sample['value'] = sigmoid(sample['value']-T.log(nce_probs))
            else:
                for sample in samples:
                    sample['value'] = sigmoid(sample['value'])
            return samples

        self.get_embed = get_embed#makes functions accessible for debug purposes
        self.get_embed_sampled = get_embed_sampled
        self.get_samples = get_samples 
        self.get_samples_logit = get_samples
        self.get_mean = get_mean_logit
        self.get_var = get_var_logit
        self.get_prob_sampled = get_prob_sampled
        self.get_prob = get_prob
        self.get_prob_logit = get_prob_logit

        if self.hypers['NCE_CORRECTION']:
            self.pos_losses = [-sample['weight']*T.mean(T.log(sample['value'])) for sample in get_samples(self.x1_emb,self.x2_emb,self.NCEprobs)]
            self.neg_losses = [-sample['weight']*T.mean(T.log(1-sample['value'])) for sample in get_samples(self.x1neg_emb,self.x2neg_emb,self.NCEprobs_neg)]
            self.bithash_1s,self.bit_p1 = get_prob_sampled(self.x1_emb,self.x2_emb,sample=False,nce_probs=self.NCEprobs)
            self.bithash_2s,self.bit_p2 = get_prob_sampled(self.x1neg_emb,self.x2neg_emb,sample=False,nce_probs=self.NCEprobs_neg)
        else:
            self.pos_losses = [-sample['weight']*T.mean(T.log(sample['value'])) for sample in get_samples(self.x1_emb,self.x2_emb)]
            self.neg_losses = [-sample['weight']*T.mean(T.log(1-sample['value'])) for sample in get_samples(self.x1neg_emb,self.x2neg_emb)]
            self.bithash_1s,self.bit_p1 = get_prob_sampled(self.x1_emb,self.x2_emb,sample=False)
            self.bithash_2s,self.bit_p2 = get_prob_sampled(self.x1neg_emb,self.x2neg_emb,sample=False)

        self.loss = sum(self.pos_losses+self.neg_losses)            
        self.sampled_loss = T.mean(-T.log(self.bit_p1)-T.log(1-self.bit_p2))
        self.samples_pos = [get_embed_sampled(self.x1_emb),get_embed_sampled(self.x2_emb)]
        self.samples_neg = [get_embed_sampled(self.x1neg_emb) ,get_embed_sampled(self.x2neg_emb)]
         
        ###################
        #construct update graph

        self.grads = T.grad(self.loss, self.params)
        if self.hypers['OPTIMIZER']=='SGD':
            self.updates_minibatch = OrderedDict([(p, p-self.lr*g/self.n_batches_train) for p,g in zip(self.params,self.grads)])
        elif self.hypers['OPTIMIZER']=='ADAGRAD':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.cast(T.switch(T.eq(g,0),p,p-(self.lr/self.n_batches_train)*g/((hg+g*g+0.000000000001)**.5)),theano.config.floatX)) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=T.cast(hg + g*g,theano.config.floatX)
        elif self.hypers['OPTIMIZER']=='RMSPROP':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.cast(T.switch(T.eq(g,0),p,p-(self.lr/self.n_batches_train)*g/((training_param*hg+(1-training_param)*g*g+0.000000000001)**.5)),theano.config.floatX)) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=T.cast(training_param*hg + (1-training_param)*g*g,theano.config.floatX)
        else:
            print 'unrecognized training algorithm'
            return -1

    def get_training_fn(self):
        #########################
        #construct optimization functions
        #########################

        #outputs
        outputs = [self.loss]
        outputnames = ['train expected loss']

        #define function
        self.train_fn = theano.function(inputs = [],
                                     outputs = outputs,
                                     updates = self.updates_minibatch,
                                     on_unused_input = 'warn')
        #extended function
        def train(index):
            #load positive samples
            idxs_batch = numpy.transpose(self.data_train[index*self.batch_size_train:(index+1)*self.batch_size_train])
            x1_idxs,x2_idxs = (idxs_batch[0],idxs_batch[1])

            #load negative samples
            x2_idxs_negative, nce_probs_neg = self.neg_sampler.sample_cdf(x1_idxs)
            nce_probs_pos = self.neg_sampler.get_prob(left_idxs=x1_idxs,right_idxs=x2_idxs)

            #set shared variables
            self.NCEprobs.set_value(nce_probs_pos)
            self.NCEprobs_neg.set_value(nce_probs_neg)
            self.x1_idxs.set_value(x1_idxs.astype(np.int32))
            self.x2_idxs.set_value(x2_idxs.astype(np.int32))
            self.x1_idxs_negative.set_value(x1_idxs.astype(np.int32))
            self.x2_idxs_negative.set_value(x2_idxs_negative.astype(np.int32))

            #return actual computation
            return self.train_fn()        

        #########################
        #Construct test functions
        #########################
        x1_idxs_validate = numpy.transpose(self.data_test)[0]
        x2_idxs_validate = numpy.transpose(self.data_test)[1]
            
        #setup validation outputs
        validate_outputs = [self.loss, self.sampled_loss]
        validate_outputnames = ['validate expected loss','validate sampled loss']

        #define theano validation function
        self.validate_fn = theano.function(inputs = [],
                                        outputs= validate_outputs)
                                        

        #define actual validation function
        def validate():
            x2_idxs_negative_validate, nce_probs_neg = self.neg_sampler.sample_cdf(x1_idxs_validate)
            nce_probs_pos = self.neg_sampler.get_prob(left_idxs=x1_idxs_validate,right_idxs=x2_idxs_validate)

            self.NCEprobs.set_value(nce_probs_pos)
            self.NCEprobs_neg.set_value(nce_probs_neg)

            self.x1_idxs.set_value(x1_idxs_validate.astype(np.int32))
            self.x2_idxs.set_value(x2_idxs_validate.astype(np.int32))
            self.x1_idxs_negative.set_value(x1_idxs_validate.astype(np.int32))
            self.x2_idxs_negative.set_value(x2_idxs_negative_validate.astype(np.int32))

            return self.validate_fn()

        
        return {'train':(train,outputnames), 'test':(validate,validate_outputnames)}

''' Testing!! '''

if __name__=='__main__':
    #data,_= pickle.load(open('/ceph/vinith/kg/embedding/data/traindata_db233_minmentions10_minentity3.pkl','r'))

    data_train = np.array([[0,1],[1,2],[2,3],[0,3],[4,5]])
    data_test = np.array([[0,2]])

    hypers_default = {'DIM':                  10,
                'N_ENTITIES':           6,
                'BATCH_SIZE':           3,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 None,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0.01,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'VALIDATION_HOLDOUT':   0.05,
                'NCE_MIX_PROB_UNIGRAM': 0.1,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        10,
    }

    # hypers_default = {'DIM':                  10,
    #             'N_ENTITIES':           6,
    #             'BATCH_SIZE':           3,
    #             'DATA_SYM_TRAINTEST':   False,
    #             'SEED':                 None,
    #             'EMBEDDING_TYPE':      'BIT',
    #             'INIT_A':               1.0,
    #             'INIT_B':               -.5,
    #             'OBJECTIVE_SAMPLES':    3,
    #             'NCE_DIST':             'GRAPH_MARKOV',
    #             'NCE_CORRECTION':       True,
    #             'NCE_B_CORRECTION':     True,
    #             'UNIGRAM_POWER':        0,
    #             'UNIGRAM_LAPLACE':      0.01,
    #             'NCE_GRAPH_DIST_POWER': 1,
    #             'NCE_GRAPH_SELF_WEIGHT':None,
    #             'NCE_GRAPH_BALANCE_1_2':0.5,
    #             'VALIDATION_HOLDOUT':   0.05,
    #             'NCE_MIX_PROB_UNIGRAM': 0.5,
    #             'OPTIMIZER':            'ADAGRAD',
    #             'LEARNING_RATE':        10,
    # }
    LP = NCE_linkpred(hypers_default,data_train,data_test)

    x = LP.get_training_fn()
    for _ in range(100):
        x['train'][0](0); x['train'][0](1)
    e = sigmoid(LP.emb).eval()
    print np.round(e.dot(e.T)+(1-e).dot((1-e).T))

    # pdb.set_trace()
    # idxs_batch = numpy.transpose(data[:10000])
    # neg_samples = gen_neg_samples(pos_samples = idxs_batch,
    #                               sampling_type = 'UNIFORM',
    #                               n_entities = max(data.flatten())+1,
    #                               cdf = None,
    #                               is_directed_prediction = False)
    # CDF = get_unigram_power(data)
    # sampledict = sample_cdf(CDF,1)




