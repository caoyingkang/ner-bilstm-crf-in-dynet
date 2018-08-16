# ner-bilstm-crf-in-dynet
An implementation of NER (Named Entity Recognition) task in dynet, using biLSTM both on character level and word level to extract informative embeddings, as well as using a linear chain CRF to calculate loss and make prediction.



In fact, this repository is a re-implementation of the model from [Guillaume Genthial](https://github.com/guillaumegenthial/sequence_tagging), who gave a tensorflow version of codes on his [github page](https://github.com/guillaumegenthial/sequence_tagging) as well as a detailed, helpful, even enlightening tutorial for overall understanding of the model on his [blog post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html). After reading his fantastic codes, an idea came to me that what if I implement the model using Dynet? As far as I know, Dynet is much younger and less mature than tensorflow, but it is said to be powerful when it comes to NLP tasks, in which data samples (sentences probably) do not have a fixed size and require special care (such as padding) to take in many neural network toolkits around, because Dynet has the capability of performing minibatching automatically. So, I tested it out and luckily, it did work! Even though the result was not bad and auto-batching did help me a lot in tackling those flexible sizes, I got countless pains in using Dynet as well: Many times I wanted to use a function or a building block to help me simplify codes, I told myself for sure that it must had been written well as a built-in utility, and then I went over to the reference manual pages and couldn't find anything! For example, since Dynet 2.0.3 didn't offer linear chain CRF functionality, I had to implement that in this model. Anyway, Dynet is indeed helpful in its own rights, and is undergoing rapid development with the help of opensource community. I am still looking forward to its success!



No more personal comments from now. In order to run codes in this repository, you will want to:

1.  install DyNet 2.0.3 for Python following instructions [here](https://dynet.readthedocs.io/en/2.0.3/python.html). 
2.  download glove word vectors using command `make glove` .(look at the `makefile` in this repository)
3.  download dataset from CoNLL2013 shared task. (For the sake of downloading correctly-formatted data as well as understanding corresponding code snippets, a small clipping of several sentences from the dataset is offered in `data/ner.en/ex.XXX.txt`)
4. run with the command `make run`.



Notes:

1. Please use DyNet 2.0.3, other versions are not promised to run correctly.

2. `make run` comprises three python executing procedures:

   - `python build_data.py`, this also comprises two stages:
     - BUILD_VOCAB
     - TRIM_DATA
   - `python train.py`
   - `python test.py`

   You can decide to run only one or some of them, but make sure that all the steps before that have been executed beforehand.

3. All the configurations of the model is placed in `model/config.py`. If you want to tune some hyperparameters, revise certain files' paths, or decide whether or not to use some building blocks, just go make changes yourself. All the item names are either self-explained or with elaborate comments. For instance:

   - `use_crf` : whether to use linear chain CRF for decoding
   - `batch_sz`: size of minibatch
   - `nepochs`: number of epochs to go
   - `dropout`: dropout probability
   - `clip_thr`: gradient clipping threshold
   - `dim_word`: dimension for word embeddings in glove
   - `dim_char`: dimension for trainable character embeddings
   - `BUILD_VOCAB`: whether to perform building vocabularies in `build_data.py`
   - `TRIM_DATA`: whether to perform trimming dataset in `build_data.py`
   - and so on ...

4. By default, it is required to run on a GPU device. To switch back to CPU, just delete following line at the very beginning of `train.py` and `test.py`:

   ```python
   dynet_config.set_gpu()
   ```

   

5. By default, auto-batching is enabled. To cancel it, just delete following line at the very beginning of `train.py` and `test.py`:

   ```python
   dynet_config.set(mem="1024", autobatch=True)
   ```

   

6. Feel free to pull requests and add comments!

