#call this version as run_experiment <experiment name>, where <experiment name> is the key in hypers.py to the experiment parameters.
from models.model_NCE import NCE_linkpred
import time
import os, shutil, sys, pdb
import cPickle as pickle
import theano, numpy, math
import numpy as np
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split

import hypers_NCE as HypersModule #local module including all user-defined hyper parameters.

#experiment parameters
#BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
#BASE_DIR = '/home/vmisra/data/kg' #piazza-watson03
# BASE_DIR = '/home/vmisra' #piazza-watson02, pok machines
# BASE_DIR = '/Users/vmisra/data/kg_embed_data' #local macbook
BASE_DIR = '/Users/vmisra/kgdata' #local macbook v2

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
    data = pickle.load(open(hypers['PATH_DATA'], 'r'))
    # data,_ = pickle.load(open(hypers['PATH_DATA'],'r'))
elif hypers['DATASET']=='WORDNET':
    data,_,_ = pickle.load(open(hypers['PATH_DATA'],'r'))
elif hypers['DATASET'] in ['SLASHDOT','FLICKR','BLOGCATALOG']:
    data = pickle.load(open(hypers['PATH_DATA'],'r'))

randstate = numpy.random.RandomState(hypers['SEED'])
data = randstate.permutation(data)

if hypers['SEED'] == None:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'])
else:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'], random_state=hypers['SEED'])

print "N edges in data: ",len(data_train)+len(data_validate)

if 'N_ENTITIES' in hypers and hypers['N_ENTITIES'] != None:
    n_entities = hypers['N_ENTITIES']
else:
    n_entities = max(data.flatten())+1
    hypers['N_ENTITIES'] = n_entities
n_batches = int(math.ceil(float(len(data_train))/float(hypers['BATCH_SIZE'])))
print "entities and batches: ",n_entities,n_batches



#init model
print "initialize model..."
LP = NCE_linkpred(hypers=hypers, data_train=data_train, data_test=data_validate)


#kickstart --- crude curriculum learning... only appropriate 
if 'KICKSTART' in hypers:
    import results.load_models_results as load_models_results
    params_old = load_models_results.copy_and_load(hypers['KICKSTART'],best_sampled=True,cache_dir =BASE_DIR)
    LP.emb.set_value(params_old[0])
    LP.a.set_value(params_old[1])
    # LP.b.set_value(params_old[2])

#initialize training
optimization_fns = LP.get_training_fn()

#########
#train loop

for epoch in range(hypers['N_EPOCHS']):
    print "Beginning epoch: ",epoch
    channels_epoch = {}
    start = time.time()
    # print 'mean NCEprob: ', np.mean(LP.NCEprobs.get_value()), np.mean(np.log(LP.NCEprobs.get_value())),np.min(LP.NCEprobs.get_value())
    # print 'mean NCEprobneg: ', np.mean(LP.NCEprobs_neg.get_value()), np.mean(np.log(LP.NCEprobs_neg.get_value())),np.min(LP.NCEprobs_neg.get_value())
    # print 'pos and neg losses', sum(LP.pos_losses).eval(),sum(LP.neg_losses).eval()
    # print "pos and neg probs, sampled:", np.mean(LP.bit_p1.eval()), np.mean(LP.bit_p2.eval())
    # print "inverted loss:", np.mean(np.log(1-LP.bit_p1.eval())+np.log(LP.bit_p2.eval()))
    # print (LP.a.get_value(),LP.b.get_value())

    for batch_count in range(n_batches):
        #train
        train_outputs = optimization_fns['train'][0](batch_count)
        for outputname,output in zip(optimization_fns['train'][1],train_outputs):
            if outputname in channels_epoch:
                channels_epoch[outputname] += output/n_batches
            else: 
                channels_epoch[outputname] = output/n_batches

    #validate
    validate_outputs= optimization_fns['test'][0]()
    for outputname,output in zip(optimization_fns['test'][1],validate_outputs):
        channels_epoch[outputname] = output

    #output results        
    monitor_string = ''
    for channel,value in channels_epoch.items():
        monitor_string = monitor_string + ' \t '+channel+':'+str(value)

    print "time: ",time.time()-start,"\t epoch: ",epoch, monitor_string


    if 'validate expected loss' in channels_epoch and (epoch == 0 or channels_epoch['validate expected loss'] < best_valid_loss):
        best_valid_loss = channels_epoch['validate expected loss']
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
