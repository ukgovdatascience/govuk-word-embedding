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

import re
import collections
import math
import random
import os
import json
import sys
import logging
import logging.config
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
from lxml import etree
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from settings import *

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('pipeline')

# Define various variables

batch_size = 128
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

# Instantiate lists and dicts to fill

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# Read in all html files in DATA_DIR

logger.info('Loading vocabulary from %s', VOCAB_FILE)

with open(VOCAB_FILE, 'r') as f:
    vocabulary = f.read()

vocabulary = tf.compat.as_str(vocabulary).split()

logger.info('Total words: %s', len(vocabulary))
logger.info('Unique words: %s', len(set(vocabulary)))

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)

logger.info('Saving reverse_dictionary to %s', REVERSE_DICT_FILE)

with open(REVERSE_DICT_FILE, 'w') as f:
    json.dump(reverse_dictionary, f)


logger.debug('Most common words (+UNK): %s', count[:5])
logger.debug('Sample data: %s ', data[:10])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, SKIP_WINDOW):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * SKIP_WINDOW
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * SKIP_WINDOW + 1  # [ SKIP_WINDOW target SKIP_WINDOW ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = SKIP_WINDOW  # target label at the center of the buffer
    targets_to_avoid = [SKIP_WINDOW]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[SKIP_WINDOW]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, SKIP_WINDOW=1)
for i in range(8):
  logger.debug('%s %s -> %s %s', batch[i], reverse_dictionary[batch[i]], 
          labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.


# Setup logging with tensorboard


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = os.path.join(OUT_DIR, "tf_logs")
logdir = "{}/run-{}/".format(root_logdir, now)


graph = tf.Graph()

with graph.as_default():

    # There are two placeholders defined here into which the data batches will
    # be passed. train_inputes is a single dimension tensor with a rowcount equal
    # the batch size, the number of columns is not specified, and so can be any
    # size. train_labels will take a similar number of rows, but is constrained
    # to a single column.

    # valid_dataset is a randomly selected subset of the top n (n = valid_window)
    # data points, and is used for the picking of comparison examples which are
    # fed back to the user during training.

  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    
      # Look up embeddings for inputs.
    embeddings = tf.Variable(
            tf.random_uniform(
            shape = [VOCABULARY_SIZE, EMBEDDING_SIZE], 
            minval = -1.0, 
            maxval = 1.0
            ))

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss. It's common practice to initiate
    # the weights with a standard deviation of 1/sqrt(n).

    nce_weights = tf.Variable(
        tf.truncated_normal(
            shape = [VOCABULARY_SIZE, EMBEDDING_SIZE],
            stddev=1.0 / math.sqrt(EMBEDDING_SIZE))
        )

    # Set the weights to zero

    nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=VOCABULARY_SIZE))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  
  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(
      input_tensor = tf.square(embeddings), 
      axis = 1, 
      keep_dims=True))

  normalized_embeddings = embeddings / norm

  # Select the random sample of examples from the embeddings that are most similar
  # to the current minibatch?

  valid_embeddings = tf.nn.embedding_lookup(
      params = normalized_embeddings, 
      ids = valid_dataset
      )
 
  # Calculate here the similarity between the embeddings and the most recent batch
  
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  loss_summary = tf.summary.scalar('LOSS', loss)
  file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# Step 5: Begin training.

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  logger.info('Tensorflow session initialized')

  average_loss = 0
  for step in xrange(NUM_STEPS):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, SKIP_WINDOW)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      logger.info('Average loss at step %s: %s', step, average_loss)
      
      summary_str = loss_summary.eval(feed_dict=feed_dict)
      
      # Save each 2000th epoch as we go along
      save_path = saver.save(session, os.path.join(MODEL_DIR, 'model.ckpt'))
      file_writer.add_summary(summary_str, step)

      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        logger.debug(log_str)
  save_path = saver.save(session, os.path.join(MODEL_DIR, 'final_model.ckpt'))
  final_embeddings = normalized_embeddings.eval()

file_writer.close()
# Step 6: Visualize the embeddings.

def save_weights(data, labels, filename='weights.csv'):
    df = pd.DataFrame(
            data = data,
            index = labels
            )
    logger.debug('Weights size is %s', df.shape)
    logger.debug('Weights df: %s', df.head())
    out_file = os.path.join(OUT_DIR, filename)
    logger.info('Saving weights to %s', out_file)
    df.to_csv(out_file)

weights_labels = [reverse_dictionary[i] for i in range(final_embeddings.shape[0])]
save_weights(final_embeddings, weights_labels)

logger.info('Finished')
