#call this version as run_experiment <experiment name>, where <experiment name> is the key in hypers.py to the experiment parameters.
from models.model_neighborhood import simple_linkpred
import time
import os, shutil, sys, pdb
import cPickle as pickle
import theano, numpy, math
import numpy as np
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import hypers as HypersModule #local module including all user-defined hyper parameters.

#experiment parameters
#BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
#BASE_DIR = '/home/vmisra/data/kg' #piazza-watson03
BASE_DIR = '/home/vmisra' #piazza-watson02, pok machines
#BASE_DIR = '/Users/vmisra/data/kg_embed_data' #local macbook

if len(sys.argv) < 2:
    print "not enough arguments"
    sys.exit(0)
#otherwise, proceed.
hypers = HypersModule.get_hypers(sys.argv[1])

#automatically-set data parameters. 
if hypers['DATASET'] == 'KG':
    if 'DB_V' not in hypers:
        hypers['DB_V'] = 233
    if 'MIN_MENTIONS' not in hypers:
        hypers['MIN_MENTIONS'] = 10
    if 'MIN_ENTITIES' not in hypers:
        hypers['MIN_ENTITIES'] = 3
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,
        'data/traindata_db%(db_v)i_minmentions%(min_mentions)i_minentity%(min_entities)i.pkl'%{'db_v':hypers['DB_V'],
                                                                                               'min_mentions':hypers['MIN_MENTIONS'],
                                                                                               'min_entities':hypers['MIN_ENTITIES']
                                                                                               })#path to KG data, should be pickled numpy array of integers, size graph_size x 2
elif hypers['DATASET'] == 'WORDNET':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/noun_relations.pkl')
elif hypers['DATASET'] == 'SLASHDOT':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/slashdot.pkl')
elif hypers['DATASET'] == 'FLICKR':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/flickr.pkl')
elif hypers['DATASET'] == 'BLOGCATALOG':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/blogcatalog.pkl')

hypers['N_EDGES'] = None #10M == full corpus for this dataset...8#

#output parameters
DIR_EXP_BASE = os.path.join(BASE_DIR,'experiments')
#hypers['SNAPSHOT_PERIOD'] = 20 #spacing of snapshot saves, in units of epochs

#paths to logs
path_thisfile = os.path.abspath(__file__)
PATH_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')
PATH_STDOUTLOG = os.path.join(DIR_EXP_BASE,hypers['EXP'],'out.log')

#set up logging
dir_exp = os.path.join(DIR_EXP_BASE,hypers['EXP'])
dir_snapshots = os.path.join(dir_exp,'snapshots')
dir_codesnaps = os.path.join(dir_exp,'codesnaps')
dir_models = os.path.join(dir_codesnaps,'models')

for dir_ in [dir_exp, dir_snapshots]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

if os.path.exists(dir_codesnaps):
    shutil.rmtree(dir_codesnaps)
os.makedirs(dir_codesnaps)
#os.makedirs(dir_models)

#first, log code files
shutil.copy(path_thisfile,dir_codesnaps)
shutil.copytree(PATH_MODELS,dir_models)

#second, log console output
import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, open(PATH_STDOUTLOG, 'w'))

#############run actual experiment
print "experiment parameters: "
for hyper in hypers.items(): print hyper

#load data
print "loading data"
if hypers['DATASET']=='KG':
    data,_ = pickle.load(open(hypers['PATH_DATA'],'r'))
elif hypers['DATASET']=='WORDNET':
    data,_,_ = pickle.load(open(hypers['PATH_DATA'],'r'))
elif hypers['DATASET'] in ['SLASHDOT','FLICKR','BLOGCATALOG']:
    data = pickle.load(open(hypers['PATH_DATA'],'r'))

#data = numpy.asarray([[0,1]])
#data = numpy.asarray([[0,1],[1,2],[3,4],[4,5],[5,6],[3,4],[3,5],[3,6]])

if hypers['N_EDGES'] != None:
    data = data[:hypers['N_EDGES']]
if hypers['SHUFFLE_DATA']:
    randstate = numpy.random.RandomState(hypers['SEED'])
    data = randstate.permutation(data)

#hypers['N_EDGES'] = len(data) #putting a ceiling on N_EDGES at the length of the dataset
if hypers['SEED'] == None:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'])
else:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'], random_state=hypers['SEED'])

if hypers['DATA_SYMMETRY']:
    for d in [data_train,data_validate]:
        d = np.concatenate([d,d[:,::-1]],axis=0)


print "N edges in data: ",len(data_train)+len(data_validate)

if 'N_ENTITIES' in hypers and hypers['N_ENTITIES'] != None:
    n_entities = hypers['N_ENTITIES']
else:
    n_entities = max(data.flatten())+1
n_batches = int(math.ceil(float(len(data_train))/float(hypers['BATCH_SIZE'])))
print "entities and batches: ",n_entities,n_batches


#init model
print "initialize model..."
LP = simple_linkpred(dim=hypers['DIM'], 
                     n_entities=n_entities,
                     batch_size=hypers['BATCH_SIZE'],
                     n_samples=hypers['N_SAMPLES'],
                     embedding_type=hypers['EMBEDDING_TYPE'],
                     parameterization=hypers['PARAMETERIZATION'],
                     init_a = hypers['INIT_A'],
                     init_b = hypers['INIT_B'],
                     init_a_n = hypers['INIT_A_N'],
                     objective_samples=hypers['OBJECTIVE_SAMPLES'],
                     neighborhood = hypers['NEIGHBORHOOD'],
                     transform_scaling = hypers['TRANSFORM_SCALING'],
                     seed = hypers['SEED'],
                     quantile_floor_and_ceiling = hypers['QUANTILE_FLOOR_AND_CEILING'],
                     graph_symmetry = hypers['GRAPH_SYMMETRY'],
                     data_symmetry = hypers['DATA_SYMMETRY'],
                     neighborhood_mean = hypers['NEIGHBORHOOD_MEAN'],
                     neighborhood_weighting = hypers['NEIGHBORHOOD_WEIGHTING'],
                     negative_sampling_type=hypers['NEGATIVE_SAMPLING_TYPE'],
                     unigram_power=hypers['UNIGRAM_POWER'],
                     directed_prediction=hypers['DIRECTED_PREDICTION'],
                     nce_correction = hypers['NCE_CORRECTION'],
                     nce_b_correction = hypers['NCE_B_CORRECTION'])

#initialize training
if 'TRAIN_PARAM' in hypers:
    train_param = float(hypers['TRAIN_PARAM'])
else:
    train_param = 0.9

optimization_fns = LP.get_training_fn(data_train.astype(numpy.int32),data_validate.astype(numpy.int32),training=hypers['TRAINING'],training_param=train_param)

#for gradient checking
update_norms = [T.sum((LP.updates_minibatch[p]-p)**2)**.5 for p in LP.params]
param_norms=[T.sum(p**2)**.5 for p in LP.params]
norm_checking = theano.function(inputs=[],
                                outputs=update_norms+param_norms,
                                givens={LP.lr:numpy.float64(hypers['LR']).astype(theano.config.floatX)})


#########
#train loop
progress_since_save = 0
for epoch in range(hypers['N_EPOCHS']):
    # pdb.set_trace()
    print LP.a.get_value(), LP.b.get_value()-hypers['INIT_B']
    # print LP.NCEprobs.get_value()
    # print LP.NCEprobs_neg.get_value()
    # LP.get_prob_sampled(LP.x1_emb,LP.x2_emb,LP.n_samples,LP.NCEprobs)[1].eval()
    # LP.get_prob_sampled(LP.x1neg_emb,LP.x2neg_emb,LP.n_samples,LP.NCEprobs_neg)[1].eval()
    # LP.get_prob_sampled(LP.x1neg_emb,LP.x2neg_emb,LP.n_samples,None)[1].eval()
    print "Beginning epoch: ",epoch
    channels_epoch = {}
    #gradient check
    # norms = norm_checking()
    # print "update norm: ", [float(n) for n in norms[:len(LP.params)]]
    # print "param norm:", [float(n) for n in norms[len(LP.params):]]
    start = time.time()
    for batch_count in range(n_batches):
        channels_batch = {}
        
        #train
        for fnname,fn,outputnames in optimization_fns:

            #setup arguments for function
            if fnname == 'train':
                args = [batch_count,hypers['LR']]
                #run function
                outputs=fn(*args)
                for outputname,output in zip(outputnames,outputs):
                    key = fnname+' '+outputname
                    channels_batch[key] = output
                    if key in channels_epoch:
                        channels_epoch[key] += output/n_batches
                    else: 
                        channels_epoch[key] = output/n_batches

    for fnname,fn,outputnames in optimization_fns:
        #setup arguments for function
        if fnname == 'validate':
            args = []
            outputs=fn(*args)
            for outputname,output in zip(outputnames,outputs):
                key = fnname+' '+outputname
                channels_epoch[key] = output
        
    monitor_string = ''
    for channel,value in channels_epoch.items():
        monitor_string = monitor_string + ' \t '+channel+':'+str(value)

    print "time: ",time.time()-start,"\t epoch: ",epoch, monitor_string


    if 'validate loss' in channels_epoch and (epoch == 0 or channels_epoch['validate loss'] < best_valid_loss):
        best_valid_loss = channels_epoch['validate loss']
        print "!!new best expectation!!"
        print "expectation: ",best_valid_loss
        LP.save(os.path.join(dir_snapshots,'params_best_exp.pkl'))

    if 'validate sampled loss' in channels_epoch and (epoch == 0 or channels_epoch['validate sampled loss'] < best_valid_sampled):
        best_valid_sampled = channels_epoch['validate sampled loss']
        print "!!new best sampled!!"
        print "sampled loss: ",best_valid_sampled        
        LP.save(os.path.join(dir_snapshots,'params_best_sampled.pkl'))

print hypers['EXP']
print "best valid loss: ", best_valid_loss
print "best valid sampled: ", best_valid_sampled
