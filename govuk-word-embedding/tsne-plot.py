# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# coding: utf-8

# Adapted from https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py

import json
import os
import logging
import logging.config
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

matplotlib.use('Agg')

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('pipeline')

# Define various variables

DATA_DIR = os.getenv('DATA_DIR')
OUT_DIR = os.getenv('OUT_DIR')
MODEL_DIR = os.path.join(OUT_DIR, 'saved_models')
VOCAB_FILE = os.getenv('VOCAB_FILE')
REVERSE_DICT_FILE = os.path.join(OUT_DIR, 'reverse_dictionary.json')
WEIGHTS_FILE = os.path.join(OUT_DIR, 'weights.csv')
PLOT_DIMS = int(os.getenv('PLOT_DIMS'))
REVERSE_DICT_FILE = os.getenv('REVERSE_DICT_FILE')

logger.info('Loading reverse_dictionary to %s', REVERSE_DICT_FILE)

with open(REVERSE_DICT_FILE, 'r') as f:
    reverse_dictionary = json.load(f)

logger.info('reverse_dictionary is of length %s', len(reverse_dictionary))

logger.info('Loading word embedding from %s', WEIGHTS_FILE)

weights = pd.read_csv(WEIGHTS_FILE)
final_embeddings = weights.iloc[:,1:].as_matrix()

def plot_with_labels(low_dim_embs, labels):
    '''
    Produce TSNE plot with labels
    '''
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    out_file = os.path.join(OUT_DIR, 'tsne.png')
    logger.info('Saving TSNE plot to %s', out_file)
    plt.savefig(out_file)

logger.info('Creating TSNE')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

logger.info('Limiting plot to %s words', PLOT_DIMS)

labels = [reverse_dictionary[str(i)] for i in range(PLOT_DIMS)]

low_dim_embs = tsne.fit_transform(final_embeddings[:PLOT_DIMS, :])

plot_with_labels(low_dim_embs, labels)

logger.info('Finished')
