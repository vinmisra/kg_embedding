from collections import OrderedDict
import sys

def get_hypers(exp):
    hypers = OrderedDict()
    #set this to true if you want to skip the SGD/ada part of training.
    hypers['SKIP_PROBABILITIES'] = True
    if exp == 'Z0':
        hypers['EXP'] = 'Z0'
        hypers['EMB_EXP'] = 'H2'
        hypers['MACHINE'] = 'macbook'
        hypers['QUANTIZATION'] = 'SIGMOID_SAMPLE'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 50
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z2':
        hypers['EXP'] = 'Z2'
        hypers['EMB_EXP'] = 'H17'
        hypers['MACHINE'] = 'piazza13'
        hypers['QUANTIZATION'] = 'SIGMOID_SAMPLE'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 5
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z1':
        hypers['EXP'] = 'Z1'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 5
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z3':
        hypers['EXP'] = 'Z3'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z4':
        hypers['EXP'] = 'Z4'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 20
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z5':
        hypers['EXP'] = 'Z5'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 50
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z6':
        hypers['EXP'] = 'Z6'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 5
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 200
    elif exp == 'Z7':
        hypers['EXP'] = 'Z7'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 20
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 200
    elif exp == 'Z8':
        hypers['EXP'] = 'Z8'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'SH'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 50
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 200
    elif exp == 'Z9':
        hypers['EXP'] = 'Z9'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 25
    elif exp == 'Z10':
        hypers['EXP'] = 'Z10'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50
    elif exp == 'Z11':
        hypers['EXP'] = 'Z11'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 100
    elif exp == 'Z12':
        hypers['EXP'] = 'Z12'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 200
    elif exp == 'Z13':
        hypers['EXP'] = 'Z13'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 2
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 2000
    elif exp == 'Z14':
        hypers['EXP'] = 'Z14'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 5
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50
    elif exp == 'Z15':
        hypers['EXP'] = 'Z15'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 10
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50
    elif exp == 'Z16':
        hypers['EXP'] = 'Z16'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 15
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50
    elif exp == 'Z17':
        hypers['EXP'] = 'Z17'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 20
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50
    elif exp == 'Z18':
        hypers['EXP'] = 'Z18'
        hypers['EMB_EXP'] = 'H35'
        hypers['MACHINE'] = 'piazza21'
        hypers['QUANTIZATION'] = 'RHP'
        hypers['DATASET'] = 'KG'
        hypers['SEED'] = 2052016
        hypers['VALIDATION_HOLDOUT'] = 0.001
        hypers['SHUFFLE_DATA'] = True
        hypers['INIT_P_WIDTH'] = 100
        hypers['ALGORITHM'] = 'ADAGRAD'
        hypers['LR'] = 10
        hypers['BATCH_SIZE'] = 100000
        hypers['N_EPOCHS'] = 10
        hypers['N_EMBED_BITS'] = 50

    else:
        print "experiment label not found! Aborting."
        sys.exit(0)
    return hypers