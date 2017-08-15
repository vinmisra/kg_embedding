from collections import OrderedDict
import sys

hypers_default = {
                'DATASET':              'KG', 
                'DIM':                  10,
                'N_ENTITIES':           None,
                'BATCH_SIZE':           100000,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    None,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       False,
                'NCE_B_CORRECTION':     False,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0.1,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'VALIDATION_HOLDOUT':   0.05,
                'NCE_MIX_PROB_UNIGRAM': 0.1,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        15,
                'N_EPOCHS':             1000,
                }
def get_hypers(exp):
    hypers = hypers_default
    if exp == 'NCE1':
        hypers.update({
                'DATASET':              'WORDNET',
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True
        })
    elif exp == 'NCE0':
        hypers = get_hypers('NCE1')
        hypers.update({                
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'DIM':                  10
        })
    elif exp == 'NCE0.5':
        hypers = get_hypers('NCE1')
        hypers.update({                
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'DIM':                  25
        })
    elif exp == 'NCE2':
        hypers = get_hypers('NCE1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0.1,
        })
    elif exp == 'NCE3':
        hypers = get_hypers('NCE1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0.5,
        })
    elif exp == 'NCE4':
        hypers = get_hypers('NCE1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      1,
        })
    elif exp == 'NCE5':
        hypers.update({
                'DATASET':              'WORDNET',
                'DIM':                  25,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True
        })
    elif exp == 'NCE6':
        hypers.update({
                'DATASET':              'WORDNET',
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'LEARNING_RATE':        15,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        .75,
                'UNIGRAM_LAPLACE':      1,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True
        })
    elif exp == 'NCE7':
        hypers=get_hypers('NCE6')
        hypers.update({
                'UNIGRAM_LAPLACE':      0,
                'NCE_MIX_PROB_UNIGRAM': 0.5,
        })
    elif exp == 'NCE8':
        hypers=get_hypers('NCE6')
        hypers.update({
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0.75
        })
    elif exp == 'NCE9':
        hypers=get_hypers('NCE6')
        hypers.update({
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0.75,
                'NCE_MIX_PROB_UNIGRAM': 0.5
        })
    elif exp == 'NCE_KG1':
        hypers.update({
                'DATASET':              'KG',
                'DIM':                  10,
                'BATCH_SIZE':           100000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
        })
    elif exp == 'NCE_KG0':
        hypers = get_hypers('NCE_KG1')
        hypers.update({
                'LEARNING_RATE':        40,
                'DIM':                  10,
            })
    elif exp == 'NCE_KG0.5':
        hypers = get_hypers('NCE_KG1')
        hypers.update({
                'LEARNING_RATE':        50,
                'DIM':                  25,
            })
    elif exp == 'NCE_KG2':
        hypers = get_hypers('NCE_KG1')
        hypers.update({       
                'DIM':                  10,
                'OBJECTIVE_SAMPLES':    None,         
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      1,
                'INIT_B':               2,
                'INIT_A':               0,
                'LEARNING_RATE':        40,
                'DATA_SYM_TRAINTEST':   False,
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'KICKSTART':            'NCE_KG1'
        })
    elif exp == 'NCE_KG2b':
        hypers = get_hypers('NCE_KG1')
        hypers.update({       
                'DIM':                  10,
                'OBJECTIVE_SAMPLES':    None,         
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      1,
                'INIT_B':               2,
                'INIT_A':               0,
                'LEARNING_RATE':        40,
                'DATA_SYM_TRAINTEST':   False,
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'KICKSTART':            'S38'
        })
    elif exp == 'NCE_KG2c':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'DIM':                  10,
                'OBJECTIVE_SAMPLES':    None,         
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      1,
                'INIT_B':               2,
                'INIT_A':               1,
                'LEARNING_RATE':        40,
                'DATA_SYM_TRAINTEST':   True,
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
        })
    elif exp == 'NCE_KG3':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      .1,
        })
    elif exp == 'NCE_KG4':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      .5,
        })
    elif exp == 'NCE_KG5':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      1,
        })
    elif exp == 'NCE_KG6':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.9,
        })
    elif exp == 'NCE_KG7':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.5,
        })
    elif exp == 'NCE_KG8':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.1,
        })
    elif exp == 'NCE_KG9':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.1,
                'KICKSTART':            'S38'
        })
    elif exp == 'NCE_KG10':
        hypers = get_hypers('NCE_KG1')
        hypers.update({                
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.5,
                'KICKSTART':            'S38'
        })
    elif exp == 'NCE_SLASHDOT1':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50
        })
    elif exp == 'NCE_SLASHDOT0':
        hypers = get_hypers('NCE_SLASHDOT1')
        hypers.update({
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'LEARNING_RATE':        50,
                'DIM':                  10,
            })
    elif exp == 'NCE_SLASHDOT0.5':
        hypers = get_hypers('NCE_SLASHDOT1')
        hypers.update({
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'LEARNING_RATE':        50,
                'DIM':                  25,
            })
    elif exp == 'NCE_SLASHDOT2':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  25,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        1,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50
        })
    elif exp == 'NCE_SLASHDOT3':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        .75,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50,
                'KICKSTART':            'DG25',
        })
    elif exp == 'NCE_SLASHDOT4':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        .5,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50,
                'KICKSTART':            'DG25',
        })
    elif exp == 'NCE_SLASHDOT5':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  25,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        .75,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50,
                'KICKSTART':            'DG30b',
        })
    elif exp == 'NCE_SLASHDOT6':
        hypers.update({
                'DATASET':              'SLASHDOT',
                'DIM':                  25,
                'BATCH_SIZE':           10000,
                'EMBEDDING_TYPE':      'BIT',
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        .5,
                'UNIGRAM_LAPLACE':      0.1,
                'DATA_SYM_TRAINTEST':   False,
                'DATA_SYM_NCE':         True,
                'LEARNING_RATE':        50,
                'KICKSTART':            'DG30b',
        })
    elif exp == 'NCE_SLASHDOT7':
        hypers = get_hypers('NCE_SLASHDOT1')
        hypers.update({
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.5,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'LEARNING_RATE':        50,
                'DIM':                  10,
            })
    elif exp == 'NCE_SLASHDOT8':
        hypers = get_hypers('NCE_SLASHDOT1')
        hypers.update({
                'NCE_DIST':             'GRAPH_MARKOV',
                'NCE_GRAPH_DIST_POWER': 0,
                'NCE_GRAPH_SELF_WEIGHT':None,
                'NCE_GRAPH_BALANCE_1_2':0.5,
                'NCE_MIX_PROB_UNIGRAM': 0.5,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'LEARNING_RATE':        50,
                'DIM':                  10,
                'KICKSTART':            'DG25'
            })
    elif exp == 'FL1':
        hypers.update({
                'DATASET':              'FLICKR', 
                'DIM':                  25,
                'BATCH_SIZE':           100000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    None,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        40,
                })
    elif exp == 'FL2':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        50,
                })
    elif exp == 'FL3':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        450,
                })
    elif exp == 'FL4':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        200,
                })
    elif exp == 'FL4b':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        200,
                'OBJECTIVE_SAMPLES':    3
                })
    elif exp == 'FL5':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        600,
                })
    elif exp == 'FL6':
        hypers = get_hypers('FL1')
        hypers.update({
                'LEARNING_RATE':        800,
                })
    elif exp == 'FL7':
        hypers.update({
                'DATASET':              'FLICKR', 
                'DIM':                  10,
                'BATCH_SIZE':           100000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        200,
                })
    elif exp == 'FL8':
        hypers = get_hypers('FL7')
        hypers.update({
                'LEARNING_RATE':        50,
                })
    elif exp == 'FL9':
        hypers = get_hypers('FL7')
        hypers.update({
                'LEARNING_RATE':        500,
                })
    elif exp == 'FL10':
        hypers.update({
                'DATASET':              'FLICKR', 
                'DIM':                  100,
                'BATCH_SIZE':           100000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'REAL',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    None,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        500,
                })
    elif exp == 'FL11':
        hypers = get_hypers('FL10')
        hypers.update({
                'LEARNING_RATE':        800,
                })
    elif exp == 'FL12':
        hypers = get_hypers('FL10')
        hypers.update({
                'LEARNING_RATE':        100,
                })
    elif exp == 'FL13':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        200,
                })
    elif exp == 'FL14':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        50,
                })
    elif exp == 'FL15':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        500,
                })
    elif exp == 'FL16':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        200,
                })
    elif exp == 'FL17':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        50,
                })
    elif exp == 'FL18':
        hypers = get_hypers('FL10')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        500,
                })
    elif exp == 'BC1':
        hypers.update({
                'DATASET':              'BLOGCATALOG', 
                'DIM':                  10,
                'BATCH_SIZE':           10000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        100,
                })
    elif exp == 'BC2':
        hypers = get_hypers('BC1')
        hypers.update({
                'LEARNING_RATE':        500
                })
    elif exp == 'BC3':
        hypers = get_hypers('BC1')
        hypers.update({
                'LEARNING_RATE':        40
                })
    elif exp == 'BC4':
        hypers = get_hypers('BC1')
        hypers.update({
                'LEARNING_RATE':        10
                })
    elif exp == 'BC5':
        hypers.update({
                'DATASET':              'BLOGCATALOG', 
                'DIM':                  25,
                'BATCH_SIZE':           10000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'BIT',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    3,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        40,
                })
    elif exp == 'BC6':
        hypers = get_hypers('BC5')
        hypers.update({
                'LEARNING_RATE':        100
                })
    elif exp == 'BC7':
        hypers = get_hypers('BC5')
        hypers.update({
                'LEARNING_RATE':        4
                })
    elif exp == 'BC8':
        hypers.update({
                'DATASET':              'BLOGCATALOG', 
                'DIM':                  100,
                'BATCH_SIZE':           10000,
                'DATA_SYM_TRAINTEST':   False,
                'SEED':                 2052016,
                'EMBEDDING_TYPE':      'REAL',
                'INIT_A':               1.0,
                'INIT_B':               -.5,
                'OBJECTIVE_SAMPLES':    None,
                'NCE_DIST':             'UNIGRAM',
                'NCE_CORRECTION':       True,
                'NCE_B_CORRECTION':     True,
                'UNIGRAM_POWER':        0,
                'UNIGRAM_LAPLACE':      0,
                'VALIDATION_HOLDOUT':   0.05,
                'OPTIMIZER':            'ADAGRAD',
                'LEARNING_RATE':        40,
                })
    elif exp == 'BC9':
        hypers = get_hypers('BC8')
        hypers.update({
                'LEARNING_RATE':        100
                })
    elif exp == 'BC10':
        hypers = get_hypers('BC8')
        hypers.update({
                'LEARNING_RATE':        4
                })
    elif exp == 'BC11':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        40
                })
    elif exp == 'BC12':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        100
                })
    elif exp == 'BC13':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  10,
                'LEARNING_RATE':        15
                })
    elif exp == 'BC14':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        40
                })
    elif exp == 'BC15':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        100
                })
    elif exp == 'BC16':
        hypers = get_hypers('BC8')
        hypers.update({
                'OBJECTIVE_SAMPLES':    3,
                'DIM':                  25,
                'LEARNING_RATE':        15
                })
    else:
        print "experiment label not found! Aborting."
        sys.exit(0)

    #defaults
    hypers['EXP'] = exp
    for label in hypers_default:
        if label not in hypers:
            hypers[label] = hypers_default[label]

    return hypers