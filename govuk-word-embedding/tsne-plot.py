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

import pandas as pd
import os
import sys
import matplotlib
import numpy as np
import logging
import logging.config
from datetime import datetime
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('pipeline')

# Define various variables

DATA_DIR = os.getenv('DATA_DIR')
OUT_DIR = os.getenv('OUT_DIR')
MODEL_DIR = os.path.join(OUT_DIR, 'saved_models')
VOCAB_FILE = os.getenv('VOCAB_FILE')
WEIGHTS_FILE = os.path.join(OUT_DIR, 'weights.csv')
plot_only = int(os.getenv('PLOT_DIMS'))

logger.info('Loading word embedding from %s', WEIGHTS_FILE)

weights = df.read_csv(WEIGHTS_FILE)


def plot_with_labels(low_dim_embs, labels, skip_window):
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

  filename = 'tsne_skips_' + str(skip_window) + '.png' 
  out_file = os.path.join(OUT_DIR, filename)
  logger.info('Saving TSNE plot to %s', out_file)
  plt.savefig(out_file)


logger.info('Creating TSNE')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

logger.info('Limiting plot to %s words', plot_only)

labels = [reverse_dictionary[i] for i in range(plot_only)]

low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

plot_with_labels(low_dim_embs, labels, skip_window = skip_window)

logger.info('Finished')
