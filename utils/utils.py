import pdb
import math
import numpy
import numpy as np
import cPickle as pickle
from collections import OrderedDict

import os
import time

from scipy.sparse import csr_matrix

'''
defines graph sparse matrix given data matrix of the usual form
'''
def get_graph(data, n_entities=None):
    if n_entities == None:
        n_entities = max(data.flatten())+1
        
    rows = np.transpose(data)[0]
    cols = np.transpose(data)[1]

    rows_sym = np.concatenate([rows,cols])
    cols_sym = np.concatenate([cols,rows])

    vals_sym = np.ones(2*len(rows))
    return csr_matrix((vals_sym,(rows_sym,cols_sym)),shape=(n_entities,n_entities))

def get_graph_mean(data, n_entities):
    rows = np.transpose(data)[0]
    cols = np.transpose(data)[1]

    rows_sym = np.concatenate([rows,cols])
    cols_sym = np.concatenate([cols,rows])
    vals_sym = np.ones(2*len(rows))

    graph = csr_matrix((vals_sym,(rows_sym,cols_sym)),shape=(n_entities,n_entities))

    degrees = 1/(graph.sum(axis=1)+0.01)
    vals_norm = np.array(degrees[rows_sym]).flatten()

    return csr_matrix((vals_norm,(rows_sym,cols_sym)),shape=(n_entities,n_entities))

def get_graph_directed(data,n_entities):
    rows = np.transpose(data)[0]
    cols = np.transpose(data)[1]
    vals = np.ones(len(rows))
    return csr_matrix((vals,(rows,cols)),shape=(n_entities,n_entities))

def get_graph_directed_mean(data,n_entities):
    rows = np.transpose(data)[0]
    cols = np.transpose(data)[1]
    vals = np.ones(len(rows))
    graph = csr_matrix((vals,(rows,cols)),shape=(n_entities,n_entities))

    degrees = 1/(graph.sum(axis=1)+0.01)
    vals_norm = np.array(degrees[rows]).flatten()

    return csr_matrix((vals_norm,(rows,cols)),shape=(n_entities,n_entities))

'''
normalizes a CSR graph, row by row.
'''
def normalize_graph_row(graph):
    #first, create normalized graph transition matrix
    degrees_row = 1.0/(np.squeeze(np.array(graph.sum(axis=1)+0.0001)))

    #normalize by degrees
    graph_T = graph.T.tocsr() #need to convert to csr to make sure indptr/indices/data are correct.
    indices_row = graph_T.indices
    indptr_col = graph_T.indptr
    reweighted_data = graph_T.data * degrees_row[indices_row]

    graph_norm = csr_matrix((reweighted_data,indices_row,indptr_col), shape=graph.shape).T.tocsr()

    return graph_norm

'''
turns a NORMALIZED markov graph (CSR) into a markov cdf
'''
def graph_pdf_to_cdf(graph_pdf):
    graph_cdf = graph_pdf.tocsr(copy=True)

    #compute "cdf": only nonzero at nonzero entries of pdf.
    #easiest way: correct for numerical overflows, then perform cumsum
    onehot_endpoints = np.zeros(len(graph_cdf.data))
    onehot_endpoints[graph_cdf.indptr[1:]-1] = 1

    probsum_endpoints = np.zeros(len(graph_cdf.data))
    probsum_endpoints[graph_cdf.indptr[1:]-1] = np.squeeze(np.array(graph_cdf.sum(axis=1)))

    graph_cdf.data += onehot_endpoints - probsum_endpoints

    #accumulate probs now that numerical errors are fixed
    graph_cdf.data = np.cumsum(graph_cdf.data)
    graph_cdf.data[graph_cdf.data %1 != 0] %= 1
    graph_cdf.data[graph_cdf.data %1 == 0] = 1

    return graph_cdf

'''
computes sampled loss using NP functionality, to get around theano bottlenecking...
ASSUMES that you are using the 'BIT" embedding type.
NEEDS TESTING BEFORE DEPLOYMENT
'''
def np_sampled_prob(x1_emb,x2_emb,a,b,nce_probs=None):
    logit = a*(x1_emb*x2_emb+(1-x1_emb)*(1-x2_emb))+b
    if nce_probs != None:
        logit -= np.log(nce_probs)
    prob = 1.0/(1+np.exp(-logit))
    return prob

def np_sampled_loss(x1_emb,x2_emb,x1neg_emb,x2neg_emb,a,b,nce_probs_pos=None,nce_probs_neg=None):
    prob_pos = np_sampled_prob(x1_emb=x1_emb,x2_emb=x2_emb,a=a,b=b,nce_probs=nce_probs_pos)
    prob_neg = np_sampled_prob(x1_emb=x1neg_emb,x2_emb=x2neg_emb,a=a,b=b,nce_probs=nce_probs_neg)

    return -np.mean(np.log(prob_pos)+np.log(1-prob_neg))

