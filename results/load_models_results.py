import os,sys,pdb
PATH_CODE = '/gsa/yktgsa/home/v/m/vmisra/kg_embedding/embedding_theano'
if PATH_CODE not in sys.path:
    sys.path.append(PATH_CODE)

import theano,numpy,sys,os
import cPickle as pickle
import math
from hypers import get_hypers
from models.model_frozen_bitembeddings import frozen_embeddings
from sklearn.cross_validation import train_test_split
from collections import OrderedDict
import numpy as np
import time

import paramiko
from scp import SCPClient

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def copy_and_load(exp,best_sampled=True,cache_dir='/ceph/vinith/kg/embedding/'):
    path_local = os.path.join(cache_dir,'deleteme_%(exp)s.pkl'%{'exp':exp})
    
    if os.path.exists(path_local):
        return pickle.load(open(path_local,'r'))['params']

    if best_sampled:
        path_remote = '/home/vmisra/experiments/%(exp)s/snapshots/params_best_sampled.pkl'%{'exp':exp}
    else:
        path_remote = '/home/vmisra/experiments/%(exp)s/snapshots/params_best_exp.pkl'%{'exp':exp}
    print 'trying to find: ',path_remote
    for machine_id in [11,12,10,13,14,16,17,18,20,21,22,23,24]:
        
        host = 'piazza%(id)i.pok.ibm.com'%{'id':machine_id}
        
        ssh = createSSHClient(host,22,'vmisra','vmisra')
        scp = SCPClient(ssh.get_transport())
        try:
            scp.get(path_remote,path_local)
            print "Machine",machine_id,"has it!"
            return pickle.load(open(path_local,'r'))['params']
        except:
            print "Machine",machine_id,"doesn't have it."
        sys.stdout.flush()
    
    return False

'''
loads deepwalk dumped files of the format "deepwalk_Slashdot_5.txt"
'''
def load_deepwalk(dataset,dim):

    if dataset.startswith('KG'): #REALLY STUPID HACK TO DEAL WITH MISMATCHED NAMING CONVENTIONS...
        dataset='KG'
    filename = "deepwalk_%(dataset)s_%(dim)i.txt" %{'dataset':dataset,'dim':dim}
    path_local = '/ceph/vinith/kg/embedding/'+filename
    path_remote = '/home/vmisra/experiments/'+filename
    machine_ids = [11,12,10,13,14,16,17,18,20,21,22,23,24]

    def read_skipgram_file(path):
        embs = numpy.loadtxt(path,dtype=float,skiprows=1,delimiter=' ')
        embs.sort(axis=0)
        return embs

    def scp_file():
        for machine_id in machine_ids:
            host = 'piazza%(id)i.pok.ibm.com'%{'id':machine_id}
            ssh = createSSHClient(host,22,'vmisra','vmisra')
            scp = SCPClient(ssh.get_transport())
            try:
                scp.get(path_remote,path_local)
                print "Machine",machine_id,"has it!"
                return
            except:
                print "Machine",machine_id,"doesn't have it."

    if not os.path.exists(path_local):
        scp_file()
    embs= read_skipgram_file(path_local)
    n_entities = np.max(embs[:,0])+1

    mat = np.random.normal(size=(n_entities,dim))
    mat[embs[:,0].astype(int),:] = embs[:,1:]
    return mat



def load_results(path_results):
    import pandas as pd
    fnames = ['Bit Embedding Experiments - DG (data gathering).csv','Bit Embedding Experiments - Observables.csv','Bit Embedding Experiments - S (sampling).csv']
    cols = ['Name','val loss expected','val loss sampled','dim','dataset','emb type','sampling']
    dfs = [pd.read_csv(os.path.join(path_results,fname)) for fname in fnames]
    return pd.concat(dfs,axis=0)[cols].set_index('Name')

