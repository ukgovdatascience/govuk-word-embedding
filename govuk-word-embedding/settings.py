import os

# generic

DATA_DIR = os.getenv('DATA_DIR')
OUT_DIR = os.getenv('OUT_DIR')

# create_vocabulary

VOCAB_FILE = os.getenv('VOCAB_FILE')

# deep learning

EMBEDDING_SIZE = int(os.getenv('EMBEDDING_DIMS'))
NUM_STEPS = int(os.getenv('NUM_STEPS'))
ROOT_LOGDIR = os.path.join(OUT_DIR, "tf_logs")
SKIP_WINDOW = int(os.getenv('SKIP_WINDOW'))
VOCABULARY_SIZE = int(os.getenv('VOCAB_SIZE'))
MODEL_DIR = os.path.join(OUT_DIR, 'saved_models')
REVERSE_DICT_FILE = os.path.join(OUT_DIR, 'reverse_dictionary.json')
WEIGHTS_FILE = os.path.join(OUT_DIR, 'word_embedding.csv')

# tsne plot

PLOT_DIMS = int(os.getenv('PLOT_DIMS'))

