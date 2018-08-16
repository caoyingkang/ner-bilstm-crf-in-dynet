import os

# unknown word/char
UNK = "-UNK-"

# not-a-named-entity tag
NONE = "O"

# command to do
BUILD_VOCAB = True
TRIM_DATA = True

# embedding dimension
dim_word  = 300 # word2vec dim
dim_char  = 100 # dim of char embedding
hdim_word = 300 # lstm dim for word
hdim_char = 100 # lstm dim for char

if not os.path.exists("tmp"):
    os.makedirs("tmp")

raw_dataset_dir = os.path.join("data", "ner.en")
raw_train_path  = os.path.join(raw_dataset_dir, "train.txt")
raw_dev_path    = os.path.join(raw_dataset_dir, "dev.txt")
raw_test_path   = os.path.join(raw_dataset_dir, "test.txt")

trim_train_path = os.path.join("tmp", "trimmed.train")
trim_dev_path   = os.path.join("tmp", "trimmed.dev")
trim_test_path  = os.path.join("tmp", "trimmed.test")

word_vocab_path = os.path.join("tmp", "word.vocab")
tag_vocab_path  = os.path.join("tmp", "tag.vocab")
char_vocab_path = os.path.join("tmp", "char.vocab")

w2v_path = os.path.join("data", "glove.6B", "glove.6B.{}d.txt".format(dim_word))

trim_w2v_path = os.path.join("tmp", "trimmed.glove.6B.{}d.npz".format(dim_word))

size_info_path = os.path.join("tmp", "size_info.json")

nepochs  = 15
batch_sz = 64
dropout  = 0.0 # if 0.0, no dropout
clip_thr = -1 # if negative, no clipping

# whether use linear chain CRF to calculate loss and predict
use_crf = True

# basename to save model, used in dy.save()
model_basename = os.path.join("tmp", "model")
