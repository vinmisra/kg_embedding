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
theano.config.compute_test_value = 'off' #'warn' #
#MODEL_DIR = '/Users/vmisra/data/kg_embed_data/'

'''
returns CDF for the unigram-to-a-power distribution of nodes, given data in an np array
'''
def get_unigram_power(data,power=1):
    counts = np.bincount(data.flatten()) #turn it into a 1d np array
    pdf = counts.astype(float)**power
    pdf = pdf/(pdf.sum())

    cdf = np.cumsum(pdf)
    return pdf,cdf

'''
given a CDF, returns a vector of n_samples sampled according to it, along with their probabilities of being sampled.
'''
def sample_cdf(cdf,n_samples,randstate = None):
    if not randstate:
        randstate = np.random.RandomState()
    samples_cdf = randstate.rand(n_samples)
    samples = np.searchsorted(cdf,samples_cdf,side='left')

    #cdf_with_zero = np.insert(cdf,0,0)
    # probs = cdf_with_zero[samples+1] - cdf_with_zero[samples]
    return samples 
    # {'samples':samples,
            # 'probs':probs}


# '''
# returns conditional pdf and cdf according to random surfer model. LOTS of zero values so be careful.
# self_path_weight: if None, keep self path at default weight. Otherwise assign it this weight pre-normalziation.
# dist_power: distribution power over second order neighborhood
# returns pdf and cdf matrices, both in CSR format.
# '''
# def get_nbhood_pdf_cdf(graph,dist_power = None,self_path_weight = None):
#     #first, create normalized graph transition matrix
#     graph_norm = utils.normalize_graph_row(graph)

#     #then propagate markov chain
#     graph_2hop = graph_norm.dot(graph_norm)

#     #adjust conditional probabilities given self_path rules
#     if self_path_weight != None:
#         graph_2hop.setdiag(self_path_weight)

#     #raise to power as prescribed...
#     if dist_power != None:
#         graph_2hop.data = graph_2hop.data**dist_power

#     #renormalize
#     if dist_power != None or self_path_weight != None:
#         graph_2hop_pdf = utils.normalize_graph_row(graph_2hop)
#     else:
#         graph_2hop_pdf = graph_2hop

#     graph_2hop_cdf = graph_2hop_pdf.tocsc().tocsr()

#     #compute "cdf": only nonzero at nonzero entries of pdf.
#     #easiest way: correct for numerical overflows, then perform cumsum
#     onehot_endpoints = np.zeros(len(graph_2hop_cdf.data))
#     onehot_endpoints[graph_2hop_cdf.indptr[1:]-1] = 1

#     probsum_endpoints = np.zeros(len(graph_2hop_cdf.data))
#     probsum_endpoints[graph_2hop_cdf.indptr[1:]-1] = np.squeeze(np.array(graph_2hop_cdf.sum(axis=1)))

#     graph_2hop_cdf.data += onehot_endpoints - probsum_endpoints

#     #accumulate probs now that numerical errors are fixed
#     graph_2hop_cdf.data = np.cumsum(graph_2hop_cdf.data)
#     graph_2hop_cdf.data[graph_2hop_cdf.data %1 != 0] %= 1
#     graph_2hop_cdf.data[graph_2hop_cdf.data %1 == 0] = 1

#     return (graph_2hop_pdf,graph_2hop_cdf)

# '''
# samples from a pdf returned by get_nbhood_pdf, samples from a pdf returned by get_unigram_power, and mixes between the two.
# returns both the samples and their probability.
# '''
# def sample_random_surfer(left_idxs,pdfmat,cdfmat,pdfvec,cdfvec,mixing_prob,randstate = None):
#     if not randstate:
#         randstate = np.random.RandomState()
#     samples_cdfspace = randstate.rand(len(left_idxs))

#     #need to use a hack to get numpy to sample from a sparse matrix of probabilities.
#     #idea: add row number to samples, and add row number to cdfs. Then insert samples into list of data points.
#     samples_cdfspace_shifted = samples_cdfspace + left_idxs

#     cdfmat = cdfmat.tocsc()
#     cdfmat.data = cdfmat.data+cdfmat.indices
#     cdfmat = cdfmat.tocsr()

#     #now insert the samples into the giant cdfmat.data incrementing array
#     right_idxs_csr = np.searchsorted(cdfmat.data,samples_cdfspace_shifted,'left')
#     right_idxs = cdfmat.indices[right_idxs_csr]

#     #extract probabilities of right_idxs
#     right_idxs_probs = pdfmat[left_idxs,right_idxs]

#     #do the same for the unigram samples
#     right_idxs_unigram = sample_cdf(cdfvec,len(left_idxs),randstate)
#     right_idxs_unigram_probs = pdfvec[right_idxs_unigram]

#     #now kith
#     mixing_probs = np.ones(len(left_idxs))*mixing_prob
#     mixing_decision = randstate.rand(mixing_probs.shape)<mixing_probs
#     mixed_samples = mixing_decision*right_idxs + (1-mixing_decision)*right_idxs_unigram
#     mixed_probs = mixing_probs*right_idxs_probs + (1-mixing_probs)*right_idxs_unigram_probs

#     return (mixed_samples,mixed_probs)


    



#setup validation inputs

''''
given:
-np 2 x n_entities array of positive samples
-sampling type (uniform or unigram)
-is_directed_prediction:
    T: swap out half of the first elements for negative samples, half of the second elements.
    F: Swap out all the second elements.
'''
def gen_neg_samples(pos_samples,sampling_type,is_directed_prediction,cdf=None, n_entities=None):
    n_samples = pos_samples.shape[1]
    #first, find replacement values
    if sampling_type=='UNIFORM':
        if not n_entities:
            print "YOU NEED TO TELL ME HOW HIGH TO SAMPLE FOR UNIFORM SAMPLING!"
            return
        replacements = np.random.randint(n_entities,size=n_samples)
    elif sampling_type=='UNIGRAM':
        if type(cdf) == type(None):
            print "YOU NEED TO GIVE ME A CDF FOR ME TO SAMPLE FROM!"
            return
        replacements = sample_cdf(cdf,n_samples)

    #next,swap in for pos_samples' values
    neg_samples = np.transpose(np.random.permutation(np.transpose(pos_samples)))

    if is_directed_prediction:
        neg_samples[1,:] = replacements
    else:
        neg_samples[0,:n_samples/2,] = replacements[:n_samples/2]
        neg_samples[1,n_samples/2:] = replacements[n_samples/2:]

    return neg_samples

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
    neighborhood: whether to use connectivity features.
    transform_scaling: whether to separately parametrize connectivity transformation scaling parameter a_n
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
                        init_a_n = 1,
                        objective_samples = None,
                        neighborhood = False,
                        transform_scaling = False,
                        quantile_floor_and_ceiling = False,
                        neighborhood_mean = False,
                        graph_symmetry = False,
                        data_symmetry = False,
                        neighborhood_weighting = False,
                        negative_sampling_type = 'UNIFORM',
                        unigram_power = 0,
                        directed_prediction= False,
                        nce_correction = False,
                        nce_b_correction = False
                        ):
        #hack to deal with legacy use of parameter, where it wasn't just a boolean.
        if neighborhood == None:
            neighborhood = False
        if neighborhood != False:
            neighborhood = True
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
        self.randstate = numpy.random.RandomState(self.seed)
        
        ##################
        ##Parameter setup
        ##################
        self.val_or_not = theano.shared(numpy.asarray(0).astype(theano.config.floatX),name='val_or_not')
        self.emb = theano.shared((self.randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX),name="emb")
        # self.emb.tag.test_value = (self.randstate.uniform(-1.0,1.0,(self.n_entities,self.dim))).astype(theano.config.floatX)

        self.a = theano.shared (numpy.asarray(self.init_a).astype(theano.config.floatX),name='a')
        if self.nce_b_correction:
            self.init_b -= np.log(self.n_entities)
        self.b = theano.shared (numpy.asarray(self.init_b).astype(theano.config.floatX),name='b')

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
        self.x1_idxs = theano.shared(numpy.asarray([0,1],dtype=numpy.int32))
        self.x2_idxs = theano.shared(numpy.asarray([2,3],dtype=numpy.int32))
        
        self.x1_idxs_negative = theano.shared(numpy.asarray([0,5],dtype=numpy.int32))
        self.x2_idxs_negative = theano.shared(numpy.asarray([2,3],dtype=numpy.int32))

        self.NCEprobs = theano.shared(numpy.asarray([.2,.2],dtype=theano.config.floatX))
        self.NCEprobs_neg = theano.shared(numpy.asarray([.2,.2],dtype=theano.config.floatX))
#define inputs
        # self.x1_idxs = T.ivector()
        # self.x2_idxs = T.ivector()
        # self.x1_idxs.tag.test_value = numpy.asarray([0,1],dtype=numpy.int32)
        # self.x2_idxs.tag.test_value = numpy.asarray([1,2],dtype=numpy.int32)

        ####################
        ### Define embedding
        ####################
        if self.neighborhood:
            self.x1_batch_neighborhood = theano.shared(csr_matrix(numpy.zeros((2,self.n_entities)).astype(theano.config.floatX))) #theano.sparse.csr_matrix(name='graph',dtype=theano.config.floatX)  #only used if neighborhood coding is used
            self.x2_batch_neighborhood = theano.shared(csr_matrix(numpy.zeros((2,self.n_entities)).astype(theano.config.floatX)))
            self.x1neg_batch_neighborhood = theano.shared(csr_matrix(numpy.zeros((2,self.n_entities)).astype(theano.config.floatX)))
            self.x2neg_batch_neighborhood = theano.shared(csr_matrix(numpy.zeros((2,self.n_entities)).astype(theano.config.floatX)))

            self.neighborhood_transform = theano.shared( numpy.eye(self.dim).astype(theano.config.floatX),name='neighborhood_transform')
            self.params += [self.neighborhood_transform]
            self.transform_scale = theano.shared(numpy.asarray(self.init_a_n).astype(theano.config.floatX),name='a_n')
            self.neighborhood_weights = theano.shared(numpy.ones(self.n_entities).astype(theano.config.floatX),name='neighborhood_weights')
            self.weighted_emb = (self.emb.T*self.neighborhood_weights).T
            if self.transform_scaling:
                self.params += [self.transform_scale]
            if self.neighborhood_weighting:
                self.params += [self.neighborhood_weights]

        def get_embed(index_tensor, secondary_index_tensor=None,batch_neighborhood=None):
            if self.neighborhood:                
                nbhood_emb = theano.sparse.structured_dot(batch_neighborhood,self.weighted_emb)

                # ######Commented out to debug, to better simulate what we're seeing in the model_test world. MUST DECOMMENT
                # if secondary_index_tensor != None:
                #     nbhood_emb -= self.emb[secondary_index_tensor].reshape((secondary_index_tensor.shape[0],self.dim))*self.val_or_not

                # if self.neighborhood_mean:
                #     nbhood_emb = nbhood_emb/(T.sum(batch_neighborhood,axis=1)-self.val_or_not+0.001)

                embed = self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim)) + self.transform_scale*T.dot(nbhood_emb,self.neighborhood_transform)
            else:
                embed = self.emb[index_tensor].reshape((index_tensor.shape[0],self.dim))

            if self.parameterization=='SIGMOID':
                return sigmoid(embed)
            elif self.parameterization=='DIRECT':
                return embed

        if self.neighborhood:
            self.x1_emb = get_embed(self.x1_idxs,self.x2_idxs,self.x1_batch_neighborhood)
            self.x2_emb = get_embed(self.x2_idxs,self.x1_idxs,self.x2_batch_neighborhood)
            self.x1neg_emb = get_embed(self.x1_idxs_negative,None,self.x1neg_batch_neighborhood)
            self.x2neg_emb = get_embed(self.x2_idxs_negative,None,self.x2neg_batch_neighborhood)
        else:
            self.x1_emb = get_embed(self.x1_idxs)
            self.x2_emb = get_embed(self.x2_idxs)
            self.x1neg_emb = get_embed(self.x1_idxs_negative)
            self.x2neg_emb = get_embed(self.x2_idxs_negative)
        

        def get_prob_logit(embed_tensor1,embed_tensor2):
            #embed_tensorX: (batch_size,dim,*)
            if self.embedding_type=='BIT':
                return self.a*T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)+self.b
            if self.embedding_type=='BIT_NCE_CORRECTED':
                return self.a*T.mean(embed_tensor1*embed_tensor2 + (1-embed_tensor1)*(1-embed_tensor2), axis=1)+self.b -np.log(self.n_entities)
            if self.embedding_type=='BIT_INTERNALB':
                return self.a*(T.mean(2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0, axis=1)+self.b) 
            if self.embedding_type=='BIT_AND':
                return self.a*T.mean(2.0*embed_tensor1*embed_tensor2, axis=1)+self.b 
            elif self.embedding_type=='REAL':   
                return self.a*T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b 
            elif self.embedding_type=='REAL_INTERNALB':   
                return self.a*(T.mean(2.0*embed_tensor1*embed_tensor2-embed_tensor1**2-embed_tensor2**2, axis=1)+self.b) 
            elif self.embedding_type=='REAL_SQRT':
                return self.a*(T.mean(1.0-(embed_tensor1-embed_tensor2)*(embed_tensor1-embed_tensor2), axis=1))**.5+self.b 
            elif self.embedding_type=='REAL_L1':
                return self.a*(T.mean(1.0-T.abs_(embed_tensor1-embed_tensor2), axis=1))+self.b 
            elif self.embedding_type=='REAL_TRAINED' or self.embedding_type=='REAL_TRAINED_L1':
                terms = [embed_tensor1*embed_tensor2, embed_tensor1, embed_tensor2,embed_tensor1**2,embed_tensor2**2]
                expr = sum([term*coef for term,coef in zip(terms,self.coefs)])
                return self.a*T.mean(expr, axis=1)+self.b

        def get_prob(embed_tensor1, embed_tensor2, nce_probs=None):
            if type(nce_probs)==type(None):
                return sigmoid(get_prob_logit(embed_tensor1,embed_tensor2))  
            else:
                return sigmoid(get_prob_logit(embed_tensor1,embed_tensor2)-T.log(nce_probs))

        def get_prob_sampled(embed_tensor1,embed_tensor2,n_samples,nce_probs=None):
            randomizationA = theano_rng.uniform(size=(embed_tensor1.shape[0],embed_tensor1.shape[1],n_samples)) #(n_batches,dim,val)
            randomizationB = theano_rng.uniform(size=(embed_tensor2.shape[0],embed_tensor2.shape[1],n_samples)) #(n_batches,dim,val)
            bithash_1 = T.switch(T.lt(randomizationA,embed_tensor1.dimshuffle(0,1,'x')),1,0) #(val,dim)
            bithash_2 = T.switch(T.lt(randomizationB,embed_tensor2.dimshuffle(0,1,'x')),1,0) #(val,dim)

            if self.nce_correction:
                return ([bithash_1,bithash_2],get_prob(bithash_1,bithash_2,nce_probs))
            else:
                return ([bithash_1,bithash_2],get_prob(bithash_1,bithash_2))

        def get_mean(embed_tensor1,embed_tensor2):
            return get_prob_logit(embed_tensor1,embed_tensor2)

        def get_var(embed_tensor1, embed_tensor2):
            p = 2.0*embed_tensor1*embed_tensor2 -embed_tensor1-embed_tensor2+1.0
            variances = p*(1-p)
            total_var = T.sum(variances,axis=1)*(self.a/T.shape(variances)[1])**2
            return total_var

        #build up list of sampling points and sampling weights, according to normal cdf approximation.
        #if objective_samples == None, stick to sub-optimal sampling scheme.
        #if objective_samples < 0, sample (-objective_samples) from a normal distribution with the approximation (monte carlo)
        def get_samples_logit(embed_tensor1,embed_tensor2):
            if self.objective_samples == None:
                return[{'weight':1.0, 'value':get_prob_logit(embed_tensor1,embed_tensor2)}]
            elif self.objective_samples > 0:
                PhiInv = lambda z: 2**.5*erfinv(2*z-1)
                means = get_mean(embed_tensor1,embed_tensor2)
                variances = get_var(embed_tensor1,embed_tensor2) + 0.00000001

                spacing = 1.0/(self.objective_samples)
                xs = []
                for i in range(self.objective_samples):
                    raw_sample = variances**.5*PhiInv(float(i+.5)*spacing)+means
                    if self.quantile_floor_and_ceiling:
                        sample = raw_sample.clip(self.a*self.b,self.a*(1+self.b))
                    else:
                        sample = raw_sample
                    xs.append({'weight':1.0*spacing,
                               'value':sample})

                # xs.append({'weight':0.5*spacing,
                #             'value':sigmoid(variances**.5*PhiInv(0.5*spacing)+means)})
                # xs.append({'weight':0.5*spacing,
                #             'value':sigmoid(variances**.5*PhiInv(1-0.5*spacing)+means)})
                return xs
            elif self.objective_samples < 0:
                normals = theano_rng.normal(size=(-self.objective_samples,embed_tensor1.shape[0]))
                means = get_mean(embed_tensor1,embed_tensor2)
                variances = get_var(embed_tensor1,embed_tensor2) + 0.00000001

                samples = normals*(variances**.5)+means ###DEBUG
                weighting = 1.0/(-self.objective_samples)

                xs=[]
                for i in range(-self.objective_samples):
                    xs.append({'weight':weighting,
                               'value': samples[i,:]})
                return xs

        def get_samples(embed_tensor1,embed_tensor2,nce_probs=None):
            samples = get_samples_logit(embed_tensor1,embed_tensor2)
            if self.nce_correction:
                for sample in samples:
                    sample['value'] = sigmoid(sample['value']-T.log(nce_probs))
            else:
                for sample in samples:
                    sample['value'] = sigmoid(sample['value'])
            return samples

        self.get_embed = get_embed#makes functions accessible for debug purposes
        self.get_samples = get_samples 
        self.get_mean = get_mean
        self.get_var = get_var
        self.get_prob_sampled = get_prob_sampled
        self.get_prob = get_prob

        self.pos_losses = [-sample['weight']*T.mean(T.log(sample['value'])) for sample in get_samples(self.x1_emb,self.x2_emb,self.NCEprobs)]
        self.neg_losses = [-sample['weight']*T.mean(T.log(1-sample['value'])) for sample in get_samples(self.x1neg_emb,self.x2neg_emb,self.NCEprobs_neg)]
        self.loss = sum(self.pos_losses+self.neg_losses)

        if self.n_samples != None:
            if self.negative_sampling_type == 'UNIFORM':
                self.bithash_1s,self.bit_p1 = get_prob_sampled(self.x1_emb,self.x2_emb,self.n_samples)
                self.bithash_2s,self.bit_p2 = get_prob_sampled(self.x1neg_emb,self.x2neg_emb,self.n_samples)
            elif self.negative_sampling_type == 'UNIGRAM':
                self.bithash_1s,self.bit_p1 = get_prob_sampled(self.x1_emb,self.x2_emb,self.n_samples,self.NCEprobs)
                self.bithash_2s,self.bit_p2 = get_prob_sampled(self.x1neg_emb,self.x2neg_emb,self.n_samples,self.NCEprobs_neg)

            self.sampled_loss = T.mean(-T.log(self.bit_p1)-T.log(1-self.bit_p2))

    def get_training_fn(self,idxs,idxs_validate=None,training='SGD',debug='None',training_param=0.9):
        ###################
        #initialize parameters
        self.pdf,self.cdf = get_unigram_power(idxs,self.unigram_power)
        

        self.lr = T.scalar('lr')
        self.lr.tag.test_value = 1
        if self.batch_size == None:
            batch_size = len(idxs)
        else:
            batch_size = self.batch_size
        n_batches = int(math.ceil(float(len(idxs))/batch_size))  

        ###################
        #initialize data 
        ##################
        #negative sampling implementation
        
        # def get_pos_samples(i):
        #     idxs_batch = numpy.transpose(idxs[i*batch_size:(i+1)*batch_size])
        #     return (idxs_batch[0],idxs_batch[1])

        # # def get_pos_samples_unigram(i):
        # #     idxs_batch = numpy.transpose(idxs[i*batch_size:(i+1)*batch_size])
        # #     return (idxs_batch[0],idxs_batch[1],pdf[idxs_batch[1]])

        # def get_neg_samples(i):
        #     idxs_neg = self.randstate.permutation(idxs[i*batch_size:(i+1)*batch_size])
        #     idxs_neg[:len(idxs_neg)/2,0] = self.randstate.randint(self.n_entities,size=len(idxs_neg)/2)
        #     idxs_neg[len(idxs_neg)/2:,1] = self.randstate.randint(self.n_entities,size=len(idxs_neg) - (len(idxs_neg)/2))
        #     idxs_neg = numpy.transpose(idxs_neg)
        #     return (idxs_neg[0],idxs_neg[1])

        ###################
        #neighborhood coding setup
        if self.neighborhood:
            rows = numpy.transpose(idxs)[0]
            cols = numpy.transpose(idxs)[1]
            if self.graph_symmetry:
                rows,cols = (numpy.concatenate([rows,cols]),numpy.concatenate([cols,rows]))
             
            vals = numpy.ones(len(rows)).astype(theano.config.floatX)
            self.graph = csr_matrix((vals, (rows,cols)),shape=(self.n_entities,self.n_entities))

            if self.neighborhood_mean:
                self.degrees = 1.0/(self.graph.sum(axis=1)+0.01)
                vals_norm = np.array(self.degrees[rows]).flatten()
                self.graph = csr_matrix((vals_norm,(rows,cols)),shape=(self.n_entities,self.n_entities))

            
        ###################
        #construct update graph
        self.grads = T.grad(self.loss, self.params)
        if training=='SGD':
            self.updates_minibatch = OrderedDict([(p, p-self.lr*g/n_batches) for p,g in zip(self.params,self.grads)])
        elif training=='ADAGRAD':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.cast(T.switch(T.eq(g,0),p,p-(self.lr/n_batches)*g/((hg+g*g+0.000000000001)**.5)),theano.config.floatX)) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=T.cast(hg + g*g,theano.config.floatX)
        elif training=='RMSPROP':
            self.history_grads = [theano.shared (numpy.zeros(shape=p.get_value().shape).astype(theano.config.floatX)) for p in self.params]
            self.updates_minibatch = OrderedDict([(p,T.cast(T.switch(T.eq(g,0),p,p-(self.lr/n_batches)*g/((training_param*hg+(1-training_param)*g*g+0.000000000001)**.5)),theano.config.floatX)) for p,g,hg in zip(self.params,self.grads,self.history_grads)])
            for hg,g in zip(self.history_grads,self.grads):
                self.updates_minibatch[hg]=T.cast(training_param*hg + (1-training_param)*g*g,theano.config.floatX)
        else:
            print 'unrecognized training algorithm'
            return -1

        #########################
        #construct optimization functions
        #########################
        training_bundle = []

        #first, training

        #inputs
        index = T.lscalar()
        inputs = [self.lr]

        #outputs
        outputs = [self.loss]
        outputnames = ['loss']
        # if self.n_samples != None:
        #     outputs += [self.sampled_loss]
        #     outputnames += ['sampled loss']
        if debug=="NORMS":
            outputs = outputs+[T.sqrt(T.sum(self.grads[0]**2)),T.sqrt(T.sum(self.emb**2))]
            outputnames += ['grad norm','embedding norm']

        #define function

        self.train = theano.function(inputs = inputs,
                                     outputs = outputs,
                                     updates = self.updates_minibatch,
                                     on_unused_input = 'warn')
        #extended function
        def train(index,lr):
            #load positive samples
            idxs_batch = numpy.transpose(idxs[index*batch_size:(index+1)*batch_size])
            x1_idxs,x2_idxs = (idxs_batch[0],idxs_batch[1])

            #load negative samples
            neg_samples = gen_neg_samples(pos_samples = idxs_batch, 
                                        sampling_type = self.negative_sampling_type, 
                                        is_directed_prediction=self.directed_prediction,
                                        cdf = self.cdf,
                                        n_entities=self.n_entities)
            x1_idxs_negative,x2_idxs_negative = (neg_samples[0],neg_samples[1])

            #set shared variables

            self.NCEprobs.set_value(self.pdf[x2_idxs])
            self.NCEprobs_neg.set_value(self.pdf[x2_idxs_negative])
            self.x1_idxs.set_value(x1_idxs)
            self.x2_idxs.set_value(x2_idxs)
            self.x1_idxs_negative.set_value(x1_idxs_negative)
            self.x2_idxs_negative.set_value(x2_idxs_negative)

            if self.neighborhood:
                x1_neighborhood = self.graph[x1_idxs]
                x2_neighborhood = self.graph[x2_idxs]
                x1neg_neighborhood = self.graph[x1_idxs_negative]
                x2neg_neighborhood = self.graph[x2_idxs_negative]
                #remove leakage
                # x1_neighborhood[range(len(x2_idxs)),x2_idxs] = 0
                # x2_neighborhood[range(len(x1_idxs)),x1_idxs] = 0

                self.x1_batch_neighborhood.set_value(x1_neighborhood)
                self.x2_batch_neighborhood.set_value(x2_neighborhood)
                self.x1neg_batch_neighborhood.set_value(x1neg_neighborhood)
                self.x2neg_batch_neighborhood.set_value(x2neg_neighborhood)
            return self.train(lr)

        training_bundle.append(('train', train, outputnames))
        

        #next, validation
        if idxs_validate != None and len(idxs_validate)>0:

            # def get_neg_samples_valid():
            #     idxs_neg_idx = self.randstate.permutation(range(len(idxs_validate)))
            #     idxs_neg = idxs_validate.copy()
            #     idxs_neg[idxs_neg_idx[:len(idxs_neg)/2],0] = self.randstate.randint(self.n_entities,size=len(idxs_neg)/2)
            #     idxs_neg[idxs_neg_idx[len(idxs_neg)/2:],1] = self.randstate.randint(self.n_entities,size=len(idxs_neg) - (len(idxs_neg)/2))
            #     idxs_neg = numpy.transpose(idxs_neg)
            #     return (idxs_neg[0],idxs_neg[1])

            # def get_neg_samples_valid_unigram():
            #     idxs_neg = self.randstate.permutation(idxs_validate)
            #     replacement_dict = sample_cdf(cdf,len(idxs_neg))
            #     idxs_neg[:len(idxs_neg)/2,0] = replacement_dict['samples'][:len(idxs_neg)/2]
            #     idxs_neg[len(idxs_neg)/2:,0] = replacement_dict['samples'][len(idxs_neg)/2:]
            #     idxs_neg = np.transpose(idxs_neg)
            #     return (idxs_neg[0],idxs_neg[1],replacement_dict['probs'])

            x1_idxs_validate = numpy.transpose(idxs_validate)[0]
            x2_idxs_validate = numpy.transpose(idxs_validate)[1]
            
            #setup validation outputs
            valid_outputs = [self.loss]
            valid_outputnames = ['loss']
            if self.n_samples != None:
                valid_outputs += [self.sampled_loss]
                valid_outputnames += ['sampled loss']

            #define theano validation function
            self.validate = theano.function(inputs = [],
                                            outputs= valid_outputs)

            #define actual validation function
            def validate():
                neg_samples = gen_neg_samples(pos_samples = numpy.transpose(idxs_validate),
                                                    sampling_type = self.negative_sampling_type, 
                                                    is_directed_prediction=self.directed_prediction,
                                                    cdf = self.cdf,
                                                    n_entities=self.n_entities)
                # elif self.negative_sampling_type=='UNIGRAM':
                #     x1_idxs_negative_validate,x2_idxs_negative_validate,probs = get_neg_samples_valid_unigram()
                #     self.neg_sampling_probs.set_value(probs)

                self.NCEprobs.set_value(self.pdf[x2_idxs_validate])
                self.NCEprobs_neg.set_value(self.pdf[neg_samples[1]])

                self.x1_idxs.set_value(x1_idxs_validate)
                self.x2_idxs.set_value(x2_idxs_validate)
                self.x1_idxs_negative.set_value(neg_samples[0])
                self.x2_idxs_negative.set_value(neg_samples[1])
                #self.val_or_not.set_value(numpy.asarray(1.0).astype(theano.config.floatX))

                if self.neighborhood:
                    x1_neighborhood = self.graph[x1_idxs_validate]
                    x2_neighborhood = self.graph[x2_idxs_validate]
                    x1neg_neighborhood = self.graph[neg_samples[0]]
                    x2neg_neighborhood = self.graph[neg_samples[1]]

                    self.x1_batch_neighborhood.set_value(x1_neighborhood)
                    self.x2_batch_neighborhood.set_value(x2_neighborhood)
                    self.x1neg_batch_neighborhood.set_value(x1neg_neighborhood)
                    self.x2neg_batch_neighborhood.set_value(x2neg_neighborhood)

                return self.validate()

            training_bundle.append(('validate', validate, valid_outputnames))
        

        #########################
        #return training bundle of form [(optimization_function, names_for_each_reported_channel)]
        #########################
        return training_bundle

''' Testing!! '''

#if __name__=='__main__':
    #data,_= pickle.load(open('/ceph/vinith/kg/embedding/data/traindata_db233_minmentions10_minentity3.pkl','r'))



    # pdb.set_trace()
    # idxs_batch = numpy.transpose(data[:10000])
    # neg_samples = gen_neg_samples(pos_samples = idxs_batch,
    #                               sampling_type = 'UNIFORM',
    #                               n_entities = max(data.flatten())+1,
    #                               cdf = None,
    #                               is_directed_prediction = False)
    # CDF = get_unigram_power(data)
    # sampledict = sample_cdf(CDF,1)




