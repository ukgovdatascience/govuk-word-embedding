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

import os
import sys
import numpy as np
from glob import glob
from lxml import etree
import re
import tensorflow as tf
import collections
import pandas as pd

import math
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import matplotlib
matplotlib.use('Agg')

BASE_DIR = '/mnt/DATA/all-of-govuk'
OUT_DIR = '/mnt/output'



texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

filenames = [y for x in os.walk(BASE_DIR) for y in glob(os.path.join(x[0], '*.html'))]

for fname in filenames:
    
    file = fname.replace(BASE_DIR,'')
    
    label_id = len(labels_index)
    labels_index[file] = label_id
    
    with open(fname, 'r', encoding = 'utf-8') as f:
        t = f.read()
        
        tree = etree.HTML(t)
        r = tree.xpath('//main//text()')
        r = ' '.join(r)
        
        # Clean the html
        
        r = r.strip().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace(',', ' ')
        
        r = r.lower()
        r = re.sub("[^a-zA-Z]"," ",r)
        r = " ".join(r.split())
        
        # Append tokens to the text list
        
        texts.append(r)
        f.close()
        labels.append(label_id)

print(labels_index)

vocabulary = " ".join(texts)

vocabulary = tf.compat.as_str(vocabulary).split()

print(vocabulary)

print('Total words: ', len(vocabulary))
print('Unique words: ', len(set(vocabulary)))
vocabulary_size = 326

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


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

# Setup logging with tensorboard

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
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
            shape = [vocabulary_size, embedding_size], 
            minval = -1.0, 
            maxval = 1.0
            ))

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss. It's common practice to initiate
    # the weights with a standard deviation of 1/sqrt(n).

    nce_weights = tf.Variable(
        tf.truncated_normal(
            shape = [vocabulary_size, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size))
        )

    # Set the weights to zero

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

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

  loss_summary = tf.summary.scalar('LOSS', loss)
  file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# Step 5: Begin training.
num_steps = 15001


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Session initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      
      summary_str = loss_summary.eval(feed_dict=feed_dict)
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
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

file_writer.close()
# Step 6: Visualize the embeddings.


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
  plt.savefig(out_file)


def save_weights(data, index, filename='weights.csv'):
    assert len(index) == low_dim_embs.shape[0]
    df = pd.DataFrame(
            data = data,
            index = labels
            )

    out_path = os.path.join(OUT_DIR, filename)
    df.to_csv(out_path)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 300
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, skip_window = skip_window)
  save_weights(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

