"""Procedure

1. extract word_vocab that appear both in dataset (train, dev, test) 
and in word2vec; also extract tag_vocab and char_vocab from training 
data; then write vocab into three files, in the format that each word/
tag/char occupies exactly one line, and in the order that more frequent 
words/chars are placed in upper lines, so that each word/tag/char may 
have an id (line #). To deal with unknown words/chars, add config.UNK 
to the first item in word_vocab/char_vocab (with id 0).
@note: this procedure is activated by config.BUILD_VOCAB
@note: id starts at zero
@TODO: remove low frequent words

2. load vocab and convert words/tags/chars in dataset (train, dev, test) 
into ids, then store the trimmed dataset for further usage. Trimmed 
dataset is formatted in such way that each line comprises three 
component (word_id, tag_id, char_ids) and sentences are separated by an 
empty line. Also, we store those word vectors that are involved in 
word_vocab into np.ndarray, of which the i-th entry corresponds to the 
i-th word, and save it as an archive.
@note: this procedure is activated by config.TRIM_DATA
@requires: procedure 1 was run previously
"""

from model import config
from model.utils import readData
from collections import Counter
import codecs
import json
import numpy as np

def build_vocab():
    '''procedure 1
    @return: None 
    '''
    # extract word_vocab in word2vec
    vocab_w2v = set()
    with codecs.open(config.w2v_path, "r", "utf-8") as f:
        for line in f:
            spl = line.strip().split()
            if len(spl) == 0: continue
            vocab_w2v.add(spl[0])
    
    # extract vocab
    word_counter = Counter()
    char_counter = Counter()
    tags = set()
    
    for word_ls, tag_ls in readData(config.raw_train_path):
        for wd in word_ls:
            if wd in vocab_w2v:
                word_counter[wd] += 1
            char_counter.update(wd)
        tags.update(tag_ls)        
    
    for word_ls, _ in readData(config.raw_dev_path):
        for wd in word_ls:
            if wd in vocab_w2v:
                word_counter[wd] += 1 

    for word_ls, _ in readData(config.raw_test_path):
        for wd in word_ls:
            if wd in vocab_w2v:
                word_counter[wd] += 1 
                
    sorted_wd_cnt  = word_counter.most_common()
    sorted_wds     = [config.UNK] + [x[0] for x in sorted_wd_cnt]
    sorted_chr_cnt = char_counter.most_common()
    sorted_chrs    = [config.UNK] + [x[0] for x in sorted_chr_cnt]
    
    # save vocab
    with codecs.open(config.word_vocab_path, "w", "utf-8") as f:
        for w in sorted_wds:
            f.write(w + "\n")
    with codecs.open(config.tag_vocab_path, "w", "utf-8") as f:
        for t in tags:
            f.write(t + "\n")
    with codecs.open(config.char_vocab_path, "w", "utf-8") as f:
        for c in sorted_chrs:
            f.write(c + "\n")


def trim_data():
    '''procedure 2
    @return: None
    '''
    # load vocab and build map: word -> id; tag -> id; char -> id
    with codecs.open(config.word_vocab_path, "r", "utf-8") as f:
        word2id = {w.strip(): id for id, w in enumerate(f)}
    with codecs.open(config.tag_vocab_path, "r", "utf-8") as f:
        tag2id  = {t.strip(): id for id, t in enumerate(f)}
    with codecs.open(config.char_vocab_path, "r", "utf-8") as f:
        char2id = {c.strip(): id for id, c in enumerate(f)}

    word_vocab_sz = len(word2id)
    tag_vocab_sz  = len(tag2id)
    char_vocab_sz = len(char2id)
    
    # trim data 
    fins  = (config.raw_train_path, config.raw_dev_path, 
            config.raw_test_path)
    fouts = (config.trim_train_path, config.trim_dev_path, 
            config.trim_test_path)
    for fin, fout in zip(fins, fouts):
        fo = codecs.open(fout, "w", "utf-8")
        for word_ls, tag_ls in readData(fin):
            for word, tag in zip(word_ls, tag_ls):
                wid  = word2id.get(word, 0)
                tid  = tag2id[tag]
                cids = [char2id.get(c, 0) for c in word]
                fo.write(str(wid) + " " + str(tid))
                for cid in cids:
                    fo.write(" " + str(cid))
                fo.write("\n")
            fo.write("\n")
        fo.close()

    # trim word2vec
    embed_mat = np.empty((word_vocab_sz, config.dim_word), dtype=np.float32)
    embed_mat[0] = np.zeros((config.dim_word,), dtype=np.float32) # embedding for unk
    with codecs.open(config.w2v_path, "r", "utf-8") as f:
        for line in f:
            spl = line.strip().split()
            if len(spl) == 0: continue
            wd = spl[0]
            if wd in word2id:
                id = word2id[wd]
                embed_mat[id] = np.asarray(spl[1:], dtype=np.float32)
    np.savez_compressed(config.trim_w2v_path, embed_mat=embed_mat)
    
    size = {"word_vocab_sz": word_vocab_sz, 
            "tag_vocab_sz": tag_vocab_sz, 
            "char_vocab_sz": char_vocab_sz}
    with open(config.size_info_path, "w") as f:
        json.dump(size, f)    


if __name__ == "__main__":
    # Procedure 1
    if config.BUILD_VOCAB:
        print("build vocab...")
        build_vocab()
        print("-done")

    # procedure 2
    if config.TRIM_DATA:
        print("trim data...")
        trim_data()
        print("-done")
        
        
        