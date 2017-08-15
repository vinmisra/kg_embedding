#call this version as run_experiment <experiment name>, where <experiment name> is the key in hypers.py to the experiment parameters.
from models.model_frozen_bitembeddings import frozen_embeddings
import time
import os, shutil, sys
import cPickle as pickle
import theano, numpy, math
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
import binary_hypers as BinaryHypersModule #local module including all user-defined hyper parameters.

#experiment parameters
#BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
#BASE_DIR = '/home/vmisra/data/kg' #piazza-watson03
BASE_DIR = '/home/vmisra' #piazza-watson02, pok machines
#BASE_DIR = '/Users/vmisra/data/kg_embed_data' #local macbook

if len(sys.argv) < 2:
    print "not enough arguments"
    sys.exit(0)
#otherwise, proceed.
hypers = BinaryHypersModule.get_hypers(sys.argv[1])

#automatically-set data parameters. 
if hypers['DATASET'] == 'KG':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/traindata_db233_minmentions10_minentity3.pkl') #path to KG data, should be pickled numpy array of integers, size graph_size x 2
elif hypers['DATASET'] == 'WORDNET':
    hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/noun_relations.pkl')

#output parameters
DIR_EXP_BASE = os.path.join(BASE_DIR,'experiments')

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

#data = numpy.asarray([[0,1]])
#data = numpy.asarray([[0,1],[1,2],[3,4],[4,5],[5,6],[3,4],[3,5],[3,6]])

if hypers['SHUFFLE_DATA']:
    randstate = numpy.random.RandomState(hypers['SEED'])
    data = randstate.permutation(data)

#separate into training/test
print "separating into train/test"
if hypers['SEED'] == None:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'])
else:
    data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'], random_state=hypers['SEED'])

print "N edges in data: ",len(data)
n_batches = int(math.ceil(float(len(data))/float(hypers['BATCH_SIZE'])))

print "batches: ",n_batches

#init model
#first, path to embedding data:
path_embedding = os.path.join(DIR_EXP_BASE,hypers['EMB_EXP'],'snapshots','params_best_sampled.pkl')
print "initialize model..."
embedding_tester = frozen_embeddings( raw_embeddings = path_embedding,
                        seed = hypers['SEED'],
                        init_p_width = hypers['INIT_P_WIDTH'],
                        quantization = hypers['QUANTIZATION'],
                        batch_size = hypers['BATCH_SIZE'],
                        n_embed_bits = hypers['N_EMBED_BITS'])


###First things first, test DCG!
print "computing DCG"
dcg = embedding_tester.test_DCG(data_validate.T[0],data_validate.T[1])
print "DCG: ", dcg

#stop here unless the hypers are telling us to train probabilities as well as compute DCG
if hypers['SKIP_PROBABILITIES']:
    sys.exit(0)

#initialize training
optimization_fns = embedding_tester.get_training_fns_logloss(idxs_train = data_train.astype(numpy.int32),
                                                             idxs_validate = data_validate.astype(numpy.int32),
                                                             algorithm = hypers['ALGORITHM'])
optimization_fns.reverse()
 #for gradient checking
# update_norms = [T.sum((embedding_tester.updates_minibatch[p]-p)**2)**.5 for p in embedding_tester.params]
# param_norms=[T.sum(p**2)**.5 for p in embedding_tester.params]
# norm_checking = theano.function(inputs=[],
#                                 outputs=update_norms+param_norms,
#                                 givens={embedding_tester.lr:numpy.float64(hypers['LR']).astype(theano.config.floatX)})
        
#train loop
for epoch in range(hypers['N_EPOCHS']):
    channels_epoch = {}

    #quick parameter check
    #print embedding_tester.p_before_sigmoid.get_value()

    for batch_count in range(n_batches):
        channels_batch = {}
        start = time.time()
    
        #train
        for fnname,fn,outputnames in optimization_fns:
            #setup arguments for function
            if fnname == 'train':
                args = [batch_count,hypers['LR']]
            elif fnname == 'validate':
                args = []
            else:
                print "unrecognized optimization function"
            #run function
            outputs=fn(*args)

            #aggregate results
            for outputname,output in zip(outputnames,outputs):
                key = fnname+' '+outputname
                channels_batch[key] = output
                if key in channels_epoch:
                    channels_epoch[key] += output/n_batches
                else: 
                    channels_epoch[key] = output/n_batches
        
    monitor_string = ''
    for channel,value in channels_epoch.items():
        monitor_string = monitor_string + ' \t '+channel+':'+str(value)

    print "time: ",time.time()-start,"\t epoch: ",epoch, monitor_string
    if 'validate loss' in channels_epoch and (epoch == 0 or channels_epoch['validate loss'] < best_valid_loss):
        best_valid_loss = channels_epoch['validate loss']
        print "!!new best expectation!!"
        print "expectation: ",best_valid_loss

    if 'validate sampled loss' in channels_epoch and (epoch == 0 or channels_epoch['validate sampled loss'] < best_valid_sampled):
        best_valid_sampled = channels_epoch['validate sampled loss']
        print "!!new best sampled!!"
        print "sampled loss: ",best_valid_sampled        

print hypers['EXP']
print "best valid loss: ", best_valid_loss
print "best valid sampled: ", best_valid_sampled
print "DCG: ", dcg
