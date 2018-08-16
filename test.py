'''procedure to test
set up:
1. use gpu
2. use autobatch
'''

import dynet_config
dynet_config.set(mem="1024", autobatch=True)
dynet_config.set_gpu()
import dynet as dy
from model import config
from model.utils import CharBiLSTMAcceptor, WordBiLSTMAcceptor, \
    evaluate, CRFAcceptor



if __name__ == "__main__":
    
    basename = config.model_basename
    
    # build a parameter collection and load params to it
    pcol = dy.ParameterCollection()
    word_embed, char_embed = dy.load(basename, pcol)
    # load biLSTMs (and crf)
    char_acceptor = CharBiLSTMAcceptor(None, None, pcol, 
                                       loadname=basename+".charBilstm")
    word_acceptor = WordBiLSTMAcceptor(None, None, None, pcol,
                                       loadname=basename+".wordBilstm")
    if config.use_crf:
        crf_acceptor = CRFAcceptor(None, pcol,
                                   loadname=basename+".crf")
    
    # evaluating on test
    if not config.use_crf:
        acc, f1 = evaluate(config.trim_test_path,
                           char_acceptor ,word_acceptor,
                           char_embed, word_embed)
    else:
        acc, f1 = evaluate(config.trim_test_path,
                           char_acceptor ,word_acceptor,
                           char_embed, word_embed, 
                           crf_acceptor)
    
    # show results
    print("evaluation on test:")
    print("acc: {}%".format(100 * acc))
    print("f1 score: {}%".format(100 * f1))
    

    
    
        