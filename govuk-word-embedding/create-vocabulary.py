
import re
import collections
import math
import random
import os
import sys
import logging
import logging.config
import numpy as np
import pandas as pd
from glob import glob
from lxml import etree
from settings import *

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('pipeline')

# Instantiate lists and dicts to fill

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# Read in all html files in DATA_DIR

logger.info('Gathering list of files to extract text from')

filenames = [y for x in os.walk(DATA_DIR) for y in glob(os.path.join(x[0], '*.html'))]

logger.info('There are %s files to read', len(filenames))

filenames_path = os.path.join(OUT_DIR, 'filenames.txt')

logger.info('Writing filenames to: %s', filenames_path)

with open(filenames_path, 'w') as f:
    for i in filenames:
        f.write("{}\n".format(i))

logger.info('filenames list written to: %s', filenames_path)


for fname in filenames:

    logger.debug('Reading %s', fname)

    file = fname.replace(DATA_DIR,'')

    label_id = len(labels_index)
    labels_index[file] = label_id
    
    logger.debug('file: %s', file)

    with open(fname, 'r', encoding = 'utf-8') as f:
        logger.debug('Extracting text from %s', fname)
        t = f.read()
        
        try:

            tree = etree.HTML(t)
            r = tree.xpath('//main//text()')
            r = ' '.join(r)
        
            # Clean the html
        
            r = r.strip().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace(',', ' ')
        
            r = r.lower()
            r = re.sub("[^a-zA-Z]"," ",r)
            r = " ".join(r.split())
        except AttributeError as ab:
                logger.exception('AttributeError while extracting text from %s: %s', fname, ab)
        except StandardError as ex:
                logger.exception('Unexpected error while extracting text from %s: %s', fname, ab)

        
        # Append tokens to the text list
        
        texts.append(r)
        f.close()
        labels.append(label_id)


vocabulary = " ".join(texts)

logger.info('Writing vocabulary to: %s', VOCAB_FILE)

with open(VOCAB_FILE, 'w') as f:
    f.write(vocabulary)

logger.info('Finished')
