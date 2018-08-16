import codecs
from model import config
import numpy as np
import dynet as dy


def readData(file):
    '''functioned as a generator: read file and yield contents by sentence
    @param file: string, path to the file
    @yield: tuple of two list, one for words in a sentence, one for tags
    @note: make sure each line in file is formatted as "word ... tag" and 
    different lines are separated by an empty line
    '''
    words = []
    tags  = []
    with codecs.open(file, "r", encoding="utf-8") as f:
        for line in f:
            spl = line.strip().split()
            if len(spl) == 0 or spl[0] == "-DOCSTART-":
                if len(words) != 0:
                    yield words, tags
                    words = []
                    tags  = []
            else:
                words.append(spl[0])
                tags.append(spl[-1])
        if len(words) != 0:
            yield words, tags
            words = []
            tags  = []


def readSample(file):
    '''functioned as a generator: read file and yield samples
    @param file: string, path to the file
    @yield: tuple of three list: ([wids], [tids], [[cids],[cids],...])
    @note: make sure each line in file is formatted as "wid tid cid cid ..." and 
    different lines are separated by an empty line
    '''
    word_ids  = []
    tag_ids   = []
    char_idss = []
    with codecs.open(file, "r", encoding="utf-8") as f:
        for line in f:
            spl = line.strip().split()
            if len(spl) == 0:
                if len(word_ids) != 0:
                    yield word_ids, tag_ids, char_idss
                    word_ids  = []
                    tag_ids   = []
                    char_idss = []
            else:
                word_ids.append(int(spl[0]))
                tag_ids.append(int(spl[1]))
                char_idss.append([int(id) for id in spl[2:]])
        if len(word_ids) != 0:
            yield word_ids, tag_ids, char_idss
            word_ids  = []
            tag_ids   = []
            char_idss = []

 
def genBatch(samples):
    '''functioned as a generator: yield batch of samples
    @param samples: list, each item is a train/dev/test sample
    @yield: a small portion of samples
    '''
    l = len(samples)
    for b in range(0, l, config.batch_sz):
        e = min(b + config.batch_sz, l - 1)
        if e > b:
            yield samples[b: e]


def logsumexp_elems(vec):
    '''elementwise log-sum-exp
    @param vec: a vector expression, dim: (d,)
    @return: a scalar expression
    '''
    dims = vec.dim()[0]
    assert len(dims) == 1, "check dimension"
    d = dims[0]
    elems = [vec[i] for i in range(d)]
    return dy.logsumexp(elems)


def logmulexp_vM(vec, Mat):
    '''calculate: log(exp(vec)*exp(A)), where * means a matrix is 
    left-multiplied by a vector
    @param vec: a vector expression, dim (d,)
    @param Mat: a matrix expression, dim (d,d)
    @return: a vector expression, dim (d,)
    '''
    vdims, Mdims = vec.dim()[0], Mat.dim()[0]
    assert len(vdims) == 1 and len(Mdims) == 2, "check dimension"
    d = vdims[0]
    Temp = dy.colwise_add(Mat, vec)
    rows = [Temp[i] for i in range(d)]
    ret = dy.logsumexp(rows)
    assert len(ret.dim()[0]) == 1, "check dimension"
    return ret
        

class CharBiLSTMAcceptor(object):
    '''An acceptor that implements the behavior of a biLSTM: given 
    a sequence of input expressions, it outputs concatenation of 
    the last hidden states of forward and backward LSTM.
    '''
    def __init__(self, indim, hdim, paramcol, loadname=None):
        '''
        @param indim: int, input dimension of biLSTM
        @param hdim: int, hidden state dimension of both forward 
        and backward LSTM
        @param paramcol: parameter collection that is to hold the 
        local parameters in biLSTM
        @param loadname: string, default=None, if it is not None, 
        load parameters instead of creating them from scratch, 
        taking loadname as the basename used in dy.load()
        '''
        if loadname is None:
            self.flstm = dy.VanillaLSTMBuilder(1, indim, hdim, paramcol)
            self.blstm = dy.VanillaLSTMBuilder(1, indim, hdim, paramcol)
            # self.flstm = dy.LSTMBuilder(1, indim, hdim, paramcol)
            # self.blstm = dy.LSTMBuilder(1, indim, hdim, paramcol)
            self.flstm.set_dropouts(config.dropout, config.dropout)
            self.blstm.set_dropouts(config.dropout, config.dropout)
        else:
            self.flstm, self.blstm = dy.load(loadname, paramcol)

        
    def __call__(self, inputs):
        '''
        @param inputs: sequence of input expressions
        @return: concatenation of the last hidden states of forward 
        and backward LSTM.
        '''
        fs = self.flstm.initial_state()
        bs = self.blstm.initial_state()
        foutputs = fs.transduce(inputs)
        boutputs = bs.transduce(list(reversed(inputs)))
        output = dy.concatenate([foutputs[-1], boutputs[-1]])
        return output
    
    def save(self, basename):
        '''save parameters, lookup parameters and builder objects to disk
        @param basename: string, file to save into, used in dy.save()
        @return: None
        '''
        obs = [self.flstm, self.blstm]
        dy.save(basename, obs)
        
    
class WordBiLSTMAcceptor(object):
    '''An acceptor that implements the behavior of a biLSTM: given a 
    sequence of input expressions, it calculates along time-dimension 
    a list of concatenations of hidden states of forward and backward 
    LSTM, then feed each item of the list into an one-layer network, 
    and finally outputs a list of corresponding scores (i.e. logits).
    '''
    def __init__(self, indim, hdim, nclass, paramcol, loadname=None):
        '''
        @param indim: int, input dimension of biLSTM
        @param hdim: int, hidden state dimension of both forward 
        and backward LSTM
        @param nclass: int, number of classes to be classified 
        @param paramcol: parameter collection that is to hold the 
        local parameters in biLSTM
        @param loadname: string, default=None, if it is not None, 
        load parameters instead of creating them from scratch, 
        taking loadname as the basename used in dy.load()
        '''
        if loadname is None:
            self.flstm = dy.VanillaLSTMBuilder(1, indim, hdim, paramcol)
            self.blstm = dy.VanillaLSTMBuilder(1, indim, hdim, paramcol)
            # self.flstm = dy.LSTMBuilder(1, indim, hdim, paramcol)
            # self.blstm = dy.LSTMBuilder(1, indim, hdim, paramcol)
            self.flstm.set_dropouts(config.dropout, config.dropout)
            self.blstm.set_dropouts(config.dropout, config.dropout)
            self.pW    = paramcol.add_parameters((nclass, 2 * hdim))
        else:
            self.flstm, self.blstm, self.pW = dy.load(loadname, paramcol)

        
    def __call__(self, inputs):
        '''
        @param inputs: sequence of input expressions
        @return: list, each item is an expression representing a vector 
        of scores (i.e. logits)
        '''
        fs = self.flstm.initial_state()
        bs = self.blstm.initial_state()
        W = self.pW.expr()
        foutputs = fs.transduce(inputs)
        boutputs = bs.transduce(list(reversed(inputs)))
        outputs = [dy.concatenate([f, b]) for f, b in 
                   zip(foutputs, list(reversed(boutputs)))]
        logits = [W * output for output in outputs]
        return logits
    
    def save(self, basename):
        '''save parameters, lookup parameters and builder objects to disk
        @param basename: string, file to save into, used in dy.save()
        @return: None
        '''
        obs = [self.flstm, self.blstm, self.pW]
        dy.save(basename, obs)
        

class CRFAcceptor(object):
    '''An acceptor that implements a linear chain CRF: given a sequence 
    of scores/logits, it predicts the tagging sequence with best global 
    score, or calculates loss based on a correct tagging sequence
    '''
    def __init__(self, nclass, paramcol, loadname=None):
        '''
        @param nclass: int, number of classes to be classified 
        @param paramcol: parameter collection that is to hold the local 
        parameters in CRF
        @param loadname: string, default=None, if it is not None, load 
        parameters instead of creating them from scratch, taking 
        loadname as the basename used in dy.load()
        '''
        if loadname is None:
            self.d  = nclass
            self.pb = paramcol.add_parameters((nclass,))
            self.pe = paramcol.add_parameters((nclass,))
            self.pT = paramcol.add_parameters((nclass, nclass))
        else:
            self.pb, self.pe, self.pT = dy.load(loadname, paramcol)
            self.d = self.pT.shape()[0]
    
    def predict(self, inputs):
        '''
        @param inputs: list, each item is an expression representing a 
        vector of scores (i.e. logits)
        @return: a list of class id
        '''
        b, e, T = self.pb.expr(), self.pe.expr(), self.pT.expr()
        l = len(inputs)
        assert l != 0, "empty input to CRF"
        if l == 1:
            scr = b + inputs[0] + e
            return [np.argmax(scr.npvalue())]
        else:
            scr = (b + inputs[0]).npvalue()
            pre_ids = []
            pre_ids.append(np.full((self.d,),-1))
            for t in range(1, l):
                # S shape: (nclass, nclass)
                # S[i,j]: accumulated score of class i at t-1, class j at t
                # note: reshape for broadcasting
                S = T.npvalue() + np.reshape(scr, (self.d, 1))
                if t == l-1:
                    S += np.reshape((inputs[t] + e).npvalue(), (1, self.d))
                else:
                    S += np.reshape(inputs[t].npvalue(), (1, self.d))
                scr = np.max(S, axis=0)
                pre_ids.append(np.argmax(S, axis=0))
            # back retrieve
            ret = []
            p = np.argmax(scr)
            for t in range(l)[::-1]:
                ret.append(p)
                p = pre_ids[t][p]
            assert p == -1
            ret.reverse()
            return ret
    
    def loss(self, inputs, golds):
        '''
        @param inputs: list, each item is an expression representing a 
        vector of scores (i.e. logits)
        @param golds: a list of gold tag ids
        @return: a scalar expression, i.e. loss
        loss = - Score(golds) + logZ
        '''
        b, e, T = self.pb.expr(), self.pe.expr(), self.pT.expr()
        l = len(inputs)
        assert l != 0, "empty input to CRF"
        assert b.dim() == inputs[0].dim(), "check dimension"
        # perform dropout
        # inputs = [dy.dropout(vec, config.dropout) for vec in inputs]
        # calculate score of golds
        scr = b[golds[0]] + e[golds[l-1]]
        for t in range(l):
            scr += inputs[t][golds[t]]
            if t > 0:
                scr += T[golds[t-1]][golds[t]]
        # calculate logZ
        vec = b + inputs[0]
        for t in range(1, l):
            vec = logmulexp_vM(vec, T) + inputs[t]
        vec += e
        logZ = logsumexp_elems(vec)
        # calculate loss
        loss = logZ - scr
        return loss

    def save(self, basename):
        '''save parameters, lookup parameters and builder objects to disk
        @param basename: string, file to save into, used in dy.save()
        @return: None
        '''
        obs = [self.pb, self.pe, self.pT]
        dy.save(basename, obs)

  
def get_chunks(tids):
    '''get a list of chunks, i.e, a list of recognized entities
    @param tids: list of tag ids (in a sentence)
    @return: set of tuples: 
        value of tuple: (chunk_type, begin, end)
        type of tuple: (string, int, int)
    '''
    with codecs.open(config.tag_vocab_path, "r", "utf-8") as f:
        tag2id = {t.strip(): id for id, t in enumerate(f)}
    with codecs.open(config.tag_vocab_path, "r", "utf-8") as f:
        id2tag = {id: t.strip() for id, t in enumerate(f)}
    assert config.NONE in tag2id, "check not-a-named-entity tag your tag vocab"
    noneid = tag2id[config.NONE]
    chunks = set()
    chunk_type, b = None, None
    for pos, tid in enumerate(tids):
        if tid == noneid:
            if chunk_type is None:
                pass
            else: # end of chunk
                chunks.add((chunk_type, b, pos))
                chunk_type, b = None, None
        else:
            tag = id2tag[tid]
            spl = tag.split("-")
            if chunk_type is None: # begin of chunk 
                # note: (consider I-XXX as B-XXX)
                chunk_type, b = spl[1], pos
            elif spl[0] == "B": # end of chunk and new begin of chunk
                chunks.add((chunk_type, b, pos))
                chunk_type, b = spl[1], pos
    if chunk_type is not None: # end of chunk    
        chunks.add((chunk_type, b, len(tids)))
    return chunks

  
def evaluate(file, char_acceptor, word_acceptor, 
             char_embed, word_embed, 
             crf_acceptor=None):
    '''evaluate performance of model on file
    @param file: string, path to test/dev file
    @param char_acceptor: CharBiLSTMAcceptor
    @param word_acceptor: WordBiLSTMAcceptor
    @param char_embed: lookup parameter
    @param word_embed: lookup parameter
    @param crf_acceptor: CRFAcceptor, default=None
    @return: float, float: accuracy, f1 score
    '''
    # extract evaluating samples
    eval_samples = list(readSample(file))
    # counts are used to calculate acc, f1
    count_tag           = 0
    count_correct_tag   = 0
    count_gold_chunk    = 0
    count_pred_chunk    = 0
    count_correct_chunk = 0
    
    num_batch = len(list(genBatch(eval_samples)))
    showpro = ShowProcess(num_batch, "evaluating done.")
    
    for batch in genBatch(eval_samples): # for each batch
        dy.renew_cg()
        logitss = [] # shape=(#sentence,#word), elem_type=expression
        goldss = [] # gold tags, shape=(#sentence,#word), elem_type=int
        for wids, tids, cidss in batch: # for each sentence
            # feed into biLSTMs and get logits
            wembeds1 = [word_embed[wid] for wid in wids]
            wembeds2 = [char_acceptor(
                            [char_embed[cid] for cid in cids]
                        ) for cids in cidss]
            wembeds = [dy.concatenate([embed1, embed2])
                       for embed1, embed2 in zip(wembeds1, wembeds2)]
            logitss.append(word_acceptor(wembeds))
            # record gold tags
            goldss.append(tids)
        # in order to use dy.forward, flatten logitss into a list of expressions
        logits_flt = []
        for logits in logitss:
            logits_flt.extend(logits)
        # do farward
        # note: no backward or update here in evaluating stage
        dy.forward(logits_flt)
        # use logits to predict
        # predss: shape=(#sentence,#word), elem_type=int
        if crf_acceptor is None: # do not use crf
            predss = [[np.argmax(vec.npvalue()) for vec in vecs] \
                     for vecs in logitss]
        else: # use crf
            predss = [crf_acceptor.predict(vecs) for vecs in logitss]
        # update counts
        for golds, preds in zip(goldss, predss): # for each sentence
            gold_chunks = get_chunks(golds)
            pred_chunks = get_chunks(preds)
            count_gold_chunk    += len(gold_chunks)
            count_pred_chunk    += len(pred_chunks)
            count_correct_chunk += len(gold_chunks & pred_chunks)
            count_tag           += len(golds)
            golds_np = np.asarray(golds, dtype=np.int8)
            preds_np = np.asarray(preds, dtype=np.int8)
            correct = golds_np == preds_np
            count_correct_tag += np.sum(correct)
            
        showpro()
        
    # calculate accuracy
    acc = float(count_correct_tag) / count_tag
    # calculate f1 score
    p  = float(count_correct_chunk) / count_pred_chunk
    r  = float(count_correct_chunk) / count_gold_chunk
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    
    return acc, f1


class ShowProcess():
    """
    tool to show progress of a procedure in training.
    effect: [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    """
    
    max_arrow = 50 # length of progress bar

    def __init__(self, max_step, infoDone = ''):
        '''
        @param max_step: int
        @param infoDone: string, info to display when done
        '''
        self.max_step = max_step
        self.infoDone = infoDone
        self.step     = 0

    def __call__(self, step=None):
        '''
        @param step: int(reset self.step) or None(advance one step)
        @return: None
        '''
        if step is not None:
            self.step = step
        else:
            self.step += 1
        num_arrow = int(self.step * self.max_arrow / self.max_step) # num of '>'
        num_line  = self.max_arrow - num_arrow # num of '-'
        percent   = self.step * 100.0 / self.max_step # percent of progress
        bar       = '[' + '>' * num_arrow + '-' * num_line + ']' + \
                    '%.2f' % percent + '%\r'
        print(bar, end='')
        if self.step >= self.max_step:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.step = 0

