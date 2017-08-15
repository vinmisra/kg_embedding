PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
import os, sys, time
if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)

import pdb
import math
import numpy
import numpy as np
import numpy.random
import cPickle as pickle
from scipy.sparse import csr_matrix

import utils

class Sampler(object):
    def __init__(self, randstate = None):
        if randstate == None:
            randstate = np.random.RandomState()
        self.randstate = randstate

    def sample_cdf(self,left_idxs_or_nsamples):
        raise NotImplementedError

    def get_prob(self,right_idxs,left_idxs):
        raise NotImplementedError


'''
Allows one to define and sample from a unigram^x distribution.
'''
class UnigramSampler(Sampler):
    def __init__(self, data, power=1, laplace = 0, randstate = None):
        super(UnigramSampler,self).__init__(randstate)
        #set internals
        self.power = power
        #find pdf/cdf
        counts = np.bincount(data[:,1].flatten())+ laplace#turn the right side into a 1d flattened array
        self.pdf = counts.astype(float)**self.power #take the power of the counts
        self.pdf = self.pdf/(self.pdf.sum()) #normalize
        self.cdf = np.cumsum(self.pdf) #and compute the cdf

    '''
    return a vector of n_samples sampled according to pdf/cdf, along with their probabilities of being sampled.
    '''
    def sample_cdf(self,left_idxs_or_nsamples):
        if type(left_idxs_or_nsamples) == type(1):
            n_samples = left_idxs_or_nsamples
        else:
            n_samples = len(left_idxs_or_nsamples)

        samples_cdf = self.randstate.rand(n_samples) #samples in cdf domain are uniform
        samples = np.searchsorted(self.cdf,samples_cdf,side='left') #search for their corresponding values using cdf
        return (samples,self.get_prob(samples))

    def get_prob(self,right_idxs,left_idxs=None):
        return np.array(self.pdf[right_idxs]).flatten()


class GraphSampler(Sampler):
    def __init__(self, graph, dist_power = None, self_path_weight = None, balance_1st_vs_2nd_degree=0.5, randstate = None):
        super(GraphSampler,self).__init__(randstate)
        #now initialize the pdf/cdf
        #first, create normalized graph transition matrix
        self.graph_norm = utils.normalize_graph_row(graph)

        #then propagate markov chain
        self.graph_2hop = self.graph_norm.dot(self.graph_norm)

        #balance first and second degree
        self.graph_2hop = balance_1st_vs_2nd_degree*self.graph_norm + (1-balance_1st_vs_2nd_degree)*self.graph_2hop

        #adjust conditional probs
        if self_path_weight != None:
            self.graph_2hop.setdiag(np.ones(graph.shape[0])*self_path_weight)

        #raise to power as prescribed
        if dist_power != None:
            self.graph_2hop.data = self.graph_2hop.data**dist_power

        #renormalize if necessary to get PDF
        if dist_power != None or self_path_weight != None:
            self.graph_2hop_pdf = utils.normalize_graph_row(self.graph_2hop)
        else:
            self.graph_2hop_pdf = self.graph_2hop
        self.graph_2hop_pdf = self.graph_2hop_pdf.tocsc().tocsr()

        #then compute CDF
        self.graph_2hop_cdf = utils.graph_pdf_to_cdf(self.graph_2hop_pdf)

        #then setup CDF for sampling purposes
        self.cdfmat = self.graph_2hop_cdf.tocsc()
        self.cdfmat.data = self.cdfmat.data+self.cdfmat.indices
        self.cdfmat = self.cdfmat.tocsr()


    '''
    return a vector of n_samples sampled according to pdf/cdf, along with their probabilities of being sampled.
    '''
    def sample_cdf(self,left_idxs):
        #need to use a hack to get numpy to sample from a sparse matrix of probabilities.
        #idea: add row number to samples, and add row number to cdfs. Then insert samples into list of data points.
        samples_cdfspace = self.randstate.rand(len(left_idxs))
        samples_cdfspace_shifted = samples_cdfspace + left_idxs

        #now insert the samples into the giant cdfmat.data incrementing array
        right_idxs_csr = np.searchsorted(self.cdfmat.data,samples_cdfspace_shifted,'left')
        right_idxs_csr = np.clip(right_idxs_csr,a_min=0,a_max=len(self.cdfmat.indices)-1)
        right_idxs = self.cdfmat.indices[right_idxs_csr]

        #extract probabilities of right_idxs
        right_idxs_probs = self.get_prob(right_idxs,left_idxs)

        return (right_idxs, right_idxs_probs)


    '''
    get probability according to transition matrix
    data assumed to be np array shape (n_samples,2)
    '''
    def get_prob(self,right_idxs,left_idxs):
        return np.array(self.graph_2hop_pdf[left_idxs,right_idxs]).flatten()



class MixtureSampler(Sampler):
    '''
    sampler_unigram: unigram sampler object
    sampler_graph: graph sampler object of some variety (e.g. GraphSampler)
    prob_unigram: probability of reverting to unigram's sampler.
    '''
    def __init__(self,unigram_sampler, graph_sampler, prob_unigram, randstate = None):
        super(MixtureSampler,self).__init__(randstate)
        self.unigram_sampler = unigram_sampler
        self.graph_sampler = graph_sampler
        self.prob_unigram = prob_unigram

    '''
    sample from the two samplers and pick the right sample.
    Return the (mixture) probability of the given sample.
    '''
    def sample_cdf(self,left_idxs):
        n_samples = len(left_idxs)

        right_idxs_unigram,_ = self.unigram_sampler.sample_cdf(n_samples)
        right_idxs_graph,_ = self.graph_sampler.sample_cdf(left_idxs)

        #now kith
        unigram_decision = self.randstate.rand(n_samples)<self.prob_unigram
        right_idxs_mixed = unigram_decision*right_idxs_unigram + (1-unigram_decision)*right_idxs_graph

        #get probabilities
        probs_mixed = self.get_prob(right_idxs_mixed,left_idxs)

        return (right_idxs_mixed,probs_mixed)

    '''
    find mixture probability for given samples
    '''
    def get_prob(self,right_idxs,left_idxs):
        dataprob_graph = self.graph_sampler.get_prob(right_idxs,left_idxs)
        dataprob_unigram = self.unigram_sampler.get_prob(right_idxs)
        return self.prob_unigram*dataprob_unigram + (1-self.prob_unigram)*dataprob_graph

if __name__ == '__main__':
    data = np.array([[0,1],[2,3]])
    data_sym = np.concatenate([data,data[:,::-1]])
    graph = utils.get_graph(data=data)

    #first test unigram sampler
    unigram_sampler = UnigramSampler(data_sym,power=0.5)
    samples,probs= unigram_sampler.sample_cdf(10)
    print samples,probs
    print unigram_sampler.get_prob(samples)

    #then test graph sampler
    graph_sampler = GraphSampler(graph=graph,dist_power=0.5, self_path_weight=0.1)
    left_idxs = np.array([0,1,2,3])
    samples,probs = graph_sampler.sample_cdf(left_idxs=left_idxs)
    print samples,probs
    print graph_sampler.get_prob(samples,left_idxs)

    #then mixture sampler
    mixture_sampler = MixtureSampler(unigram_sampler=unigram_sampler,
                                     graph_sampler = graph_sampler,
                                     prob_unigram = 0.5)
    samples,probs = mixture_sampler.sample_cdf(left_idxs=left_idxs)
    print samples,probs
    print mixture_sampler.get_prob(samples,left_idxs)