'''procedure to train
set up:
1. use gpu
2. use autobatch
'''

import dynet_config
dynet_config.set(mem="1024", autobatch=True)
dynet_config.set_gpu()
import dynet as dy
from model import config
from model.utils import readSample, genBatch, CharBiLSTMAcceptor, \
    WordBiLSTMAcceptor, evaluate, CRFAcceptor, ShowProcess
import numpy as np
import json
import random
import time


if __name__ == "__main__":

    # extract vocab size info
    with open(config.size_info_path) as f:
        size_info = json.load(f)
    char_vocab_sz = size_info["char_vocab_sz"]
    tag_vocab_sz  = size_info["tag_vocab_sz"]
    # extract training samples
    train_samples = list(readSample(config.trim_train_path))
    # build a parameter collection and add lookup params to it
    pcol = dy.ParameterCollection()
    word_embed = pcol.lookup_parameters_from_numpy(
        np.load(config.trim_w2v_path)["embed_mat"], name="wordEmbed")
    char_embed = pcol.add_lookup_parameters(
        (char_vocab_sz, config.dim_char), name="charEmbed")
    # set a trainer
    trainer = dy.AdamTrainer(pcol)
    trainer.set_clip_threshold(config.clip_thr)
    # instantiate biLSTMs (and crf)
    char_acceptor = CharBiLSTMAcceptor(config.dim_char, config.hdim_char, pcol)
    word_acceptor = WordBiLSTMAcceptor(config.dim_word + 2 * config.hdim_char, 
                                       config.hdim_word, 
                                       tag_vocab_sz, 
                                       pcol)
    if config.use_crf:
        crf_acceptor = CRFAcceptor(tag_vocab_sz, pcol)
    # training
    for epoch in range(config.nepochs): # for each iter of training data
        print("\nepoch{}...".format(epoch + 1))
        time1 = time.time()
        random.shuffle(train_samples)
        batch_loss_ls = []
        
        num_batch = len(list(genBatch(train_samples)))
        showpro   = ShowProcess(num_batch, "training done.")
        
        for batch in genBatch(train_samples): # for each batch
            dy.renew_cg()
            snt_loss_ls = []
            
            for wids, tids, cidss in batch: # for each sentence
                # feed into biLSTMs and get logits
                wembeds1 = [dy.lookup(word_embed, index=wid, update=False)
                            for wid in wids] # TODO
                wembeds2 = [char_acceptor(
                                [char_embed[cid] for cid in cids]
                            ) for cids in cidss]
                wembeds  = [dy.concatenate([embed1, embed2]) 
                           for embed1, embed2 in zip(wembeds1, wembeds2)]
                logits = word_acceptor(wembeds)
                # calculate loss of this sentence
                if not config.use_crf: # use softmax
                    word_loss_ls = [dy.pickneglogsoftmax(vec, tid) 
                               for vec, tid in zip(logits, tids)]
                    snt_loss = dy.esum(word_loss_ls)
                else: # use linear chain crf
                    snt_loss = crf_acceptor.loss(logits, tids)
                snt_loss_ls.append(snt_loss)
            # calculate loss of this batch
            batch_loss = dy.esum(snt_loss_ls) / len(batch)
            # do forward, backward and update parameters
            batch_loss_value = batch_loss.value()
            batch_loss.backward()
            trainer.update()
            batch_loss_ls.append(batch_loss_value)
            
            showpro()
            
        total_loss = sum(batch_loss_ls) / len(batch_loss_ls)
        time2 = time.time()
        # print info
        print("total loss: {}".format(total_loss))
        print("time consumed for training: {}s".format(time2 - time1))
        # showing evaluation on dev
        if not config.use_crf:
            acc, f1 = evaluate(config.trim_dev_path,
                               char_acceptor ,word_acceptor,
                               char_embed, word_embed)
        else:
            acc, f1 = evaluate(config.trim_dev_path,
                               char_acceptor ,word_acceptor,
                               char_embed, word_embed, 
                               crf_acceptor)
        time3 = time.time()
        # print info
        print("acc: {}%\nf1 score: {}%".format(100 * acc, 100 * f1))
        print("time consumed for evaluating: {}s".format(time3 - time2))
        print("epoch done.")
        
    # save model parameters, lookup parameters and builder objects to disk
    obs = [word_embed, char_embed]
    basename = config.model_basename
    dy.save(basename, obs)
    char_acceptor.save(basename + ".charBilstm")
    word_acceptor.save(basename + ".wordBilstm")
    if config.use_crf:
        crf_acceptor.save(basename + ".crf")

    