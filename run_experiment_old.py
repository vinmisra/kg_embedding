from models.model_sampling import simple_linkpred
import time
import os, shutil
import cPickle as pickle
import theano, numpy, math
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split

#experiment parameters
#BASE_DIR = '/ceph/vinith/kg/embedding/' #piazza-vm03
#BASE_DIR = '/home/vmisra/data/kg' #piazza-watson03
BASE_DIR = '/home/vmisra' #piazza-watson02, pok machines

hypers = OrderedDict()
hypers['EXP'] = 'E-3' #exp identifier
hypers['DIM'] = 25   #embedding dimensionality
hypers['N_ENTITIES'] = None
hypers['BATCH_SIZE'] = 100000 #batch size for training
hypers['N_SAMPLES'] = 1 #number of samples to take while computing sampled loss. relevant for reported performance, but not for gradients.
hypers['OBJECTIVE_SAMPLES'] = 1
hypers['VALIDATION_HOLDOUT'] = 0.001 #fraction of edges to hold out for validation purposes
hypers['LR'] = 30  #learning rate for training
hypers['TRAINING'] = 'ADAGRAD' #either 'ADAGRAD' or 'SGD'
hypers['EMBEDDING_TYPE'] = 'BIT_INTERNALB'
hypers['PARAMETERIZATION'] = 'SIGMOID'
hypers['N_EPOCHS']=200
hypers['INIT_A'] = hypers['DIM']**.5#1.0
hypers['INIT_B'] = -.5#0.0

#data parameters
hypers['PATH_DATA'] = os.path.join(BASE_DIR,'data/traindata_db233_minmentions10_minentity3.pkl') #path to KG data, should be pickled numpy array of integers, size graph_size x 2
hypers['N_EDGES'] = None #10M == full corpus for this dataset...

#output parameters
DIR_EXP_BASE = os.path.join(BASE_DIR,'experiments')
hypers['SNAPSHOT_PERIOD'] = 20 #spacing of snapshot saves, in units of epochs

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
data,entitymap = pickle.load(open(hypers['PATH_DATA'],'r'))
#data = numpy.asarray([[0,1],[1,2],[3,4],[4,5],[5,6],[3,4],[3,5],[3,6]])
if hypers['N_EDGES'] != None:
    data = data[:hypers['N_EDGES']]
#hypers['N_EDGES'] = len(data) #putting a ceiling on N_EDGES at the length of the dataset
data_train, data_validate = train_test_split(data,test_size = hypers['VALIDATION_HOLDOUT'])
print "N edges in data: ",len(data)

#separate into training/test
print "separating into train/test"

if 'N_ENTITIES' in hypers and hypers['N_ENTITIES'] != None:
    n_entities = hypers['N_ENTITIES']
else:
    n_entities = max(data.flatten())+1
n_batches = int(math.ceil(float(len(data))/float(hypers['BATCH_SIZE'])))
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
                     objective_samples=hypers['OBJECTIVE_SAMPLES'])

#initialize training
optimization_fns = LP.get_training_fn(data_train.astype(numpy.int32),data_validate.astype(numpy.int32),hypers['TRAINING']) #WARNING: doesn't yet have support for validation...

#for gradient checking
emb_update_norm = T.sum((LP.updates_minibatch[LP.emb]-LP.emb)**2)**.5
emb_norm=T.sum(LP.emb**2)**.5
# norm_checking = theano.function(inputs=[],
#                                 outputs=[emb_update_norm,emb_norm],
#                                 givens={LP.x1_idxs: numpy.transpose(data_train[:hypers['BATCH_SIZE']].astype(numpy.int32))[0],
#                                                 LP.x2_idxs: numpy.transpose(data_train[:hypers['BATCH_SIZE']].astype(numpy.int32))[1],
#                                                 LP.lr:float(hypers['LR'])})
        
#train loop
progress_since_save = 0
for epoch in range(hypers['N_EPOCHS']):
    start = time.time()
    channels = {}

    #gradient check
    # update_norm,param_norm = norm_checking()
    # print "update norm: ", update_norm
    # print "emb norm:", param_norm

    for batch_count in range(n_batches):
        #save
        progress_since_save += 1.0/n_batches
        if progress_since_save > hypers['SNAPSHOT_PERIOD']:
            print "save point: batch ",batch_count,"\t epoch ",epoch
            save_path = os.path.join(dir_snapshots,'params_'+str(epoch)+'_'+str(batch_count)+'.pkl')
            LP.save(save_path)
            progress_since_save = 0
        
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
                if key in channels:
                    channels[key] += output/n_batches
                else: 
                    channels[key] = output/n_batches
    
    monitor_string = ''
    for channel,value in channels.items():
        monitor_string = monitor_string + ' \t '+channel+':'+str(value)

    print "time: ",time.time()-start,"\t epoch: ",epoch,monitor_string

    if 'validate loss' in channels and (epoch == 0 or channels['validate loss'] < best_valid_loss):
        print "!!new best validation!!"
        best_valid_loss = channels['validate loss']
        LP.save(os.path.join(dir_snapshots,'params_best.pkl'))

print hypers['EXP']
