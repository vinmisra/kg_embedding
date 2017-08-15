#call this version as run_experiment <experiment name>, where <experiment name> is the key in hypers.py to the experiment parameters.
from models.model_frozen_bitembeddings import frozen_embeddings
import time
import os, shutil, sys
import cPickle as pickle
import theano, numpy, math
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import hypers as HypersModule

#experiment parameters
#BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
#BASE_DIR = '/home/vmisra/data/kg' #piazza-watson03
#BASE_DIR = '/home/vmisra' #piazza-watson02, pok machines
BASE_DIR = '/Users/vmisra/data/kg_embed_data' #local macbook

#get list of experiments on this machine
DIR_EXP_BASE = os.path.join(BASE_DIR,'experiments')
exps = [o for o in os.listdir(DIR_EXP_BASE) if os.path.isdir(os.path.join(DIR_EXP_BASE,o)) and (o.startswith('H') or o.startswith('I'))]
print exps


data_dict = {}
dcg = {}
for exp in exps:
    print "starting experiment ", exp
    hypers = HypersModule.get_hypers(exp)

    #load data
    print "loading and shuffling data"

    if hypers['DATASET']=='KG' and 'KG' not in data_dict:
        dat_path = os.path.join(BASE_DIR,'data/traindata_db233_minmentions10_minentity3.pkl') #path to KG data, should be pickled numpy array of integers, size graph_size x 2
        data_dict['KG'],_ = pickle.load(open(dat_path,'r'))
    elif hypers['DATASET']=='WORDNET' and 'WORDNET' not in data_dict:
        hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/noun_relations.pkl')
        data_dict['WORDNET'],_,_ = pickle.load(open(dat_path,'r'))

    if hypers['SHUFFLE_DATA']:
        randstate = numpy.random.RandomState(hypers['SEED'])
        data = randstate.permutation(data_dict[hypers['DATASET']])
    else:
        data = data_dict[hypers['DATASET']]

    print "separating into train/test"
    if hypers['SEED'] == None:
        data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'])
    else:
        data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'], random_state=hypers['SEED'])

    #load model
    print "load model"
    path_embedding = os.path.join(DIR_EXP_BASE,exp,'snapshots','params_best_sampled.pkl')
    path_embedding_fallback = os.path.join(DIR_EXP_BASE,exp,'snapshots','params_best.pkl')

    if not os.path.exists(path_embedding):
        print 'Sampled embedding not found for experiment: ', exp, '; going to fallback.'
        if not os.path.exists(path_embedding_fallback):
            print 'No embedding found for experiment. going to next experiment'
            continue
        path_embedding = path_embedding_fallback
    
    #compute DCG
    embedding_tester = frozen_embeddings(   raw_embeddings = path_embedding,
                                            seed = hypers['SEED'],
                                            init_p_width = 2,
                                            quantization = 'SIGMOID_SAMPLE',
                                            batch_size = hypers['BATCH_SIZE'],
                                            n_embed_bits = hypers['DIM']) #not using most of these arguments, since we aren't training the log loss model here.

    
    print "computing DCG"
    dcg[exp] = embedding_tester.test_DCG(data_validate.T[0],data_validate.T[1])

    print "dcg for exp ",exp," is: ",dcg[exp]


print "all together now:"
for exp in dcg:
    print exp, dcg