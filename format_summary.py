#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import re
import pdb
from collections import defaultdict

datasets = ['train', 'val', 'test']

def LoadTokenMapping(filename):
  """Loads a token mapping from the given filename.
  Args:
    filename: The filename containing the token mapping.
  Returns:
    A list of (start, end) where start and
    end (inclusive) are offsets into the content for a token. The list is
    sorted.
  """

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for token_mapping in line.split(';'):
      if not token_mapping:
        continue

      start, length = token_mapping.split(',')

      mapping.append((int(start), int(start) + int(length)))

    mapping.sort(key=lambda x: x[1])  # Sort by start.

  return mapping


def Tokenize(text, url_hash, tokens_path):
  """Tokenizes a news story.
  Args:
    story: The Story.
    corpus: The corpus of the news story.
  Returns:
    A TokenizedStory containing the URL and the tokens or None if no token
    mapping was found for the URL.
  """

  mapping_filename = '%s/%s.txt' % (tokens_path, url_hash)
  if not os.path.exists(mapping_filename):
    return None

  mapping = LoadTokenMapping(mapping_filename)

  tokens = []
  for (start, end) in mapping:
    tokens.append(text[start:end + 1])
    if end+1 >= len(text) or text[end+1] == '\n':
      tokens.append('</s>')

  return tokens



# note: kind of specific to CNN+Dailymail
def format(path, outfile, dataset):
    print "Processing dataset:", dataset
    story_path = '%s/stories/%s' % (path, dataset)
    tokens_path = '%s/tokens' % path
    files = ['%s/%s' % (story_path, x) for x in os.listdir(story_path)]
    counter = 0

    src_file = open("%s-src-%s.txt" % (outfile, dataset), "w")
    targ_file = open("%s-targ-%s.txt" % (outfile, dataset), "w")
    for f in files:
        with open(f, "r") as fh:
            url_hash = f.split('/')[-1].split('.')[-2]
            text = fh.read()
            tokens = Tokenize(text, url_hash, tokens_path)
            assert tokens, 'tokenization error!'

            # lowercase and replace numbers
            parts = ' '.join(tokens).lower()
            parts = re.sub(r"\d", "#", parts)
            parts = parts.split('@ highlight </s>')
            src = parts[0]
            targ = parts[1].replace('</s>', '').strip()  # TODO: consider making all highlights relevant

            # print url_hash
            # print src
            # print targ
            # raw_input()
            src_file.write(src + '\n')
            targ_file.write(targ + '\n')
            counter += 1
            if counter % 10000 == 0:
                print "Processed", counter

    src_file.close()
    targ_file.close()

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcdir', help="Path to source training data. ", type=str, required=True)
    parser.add_argument('--outfile', help="Prefix of the output file names. ", type=str, required=True)
    args = parser.parse_args(arguments)

    src_dir = args.srcdir
    outfile = args.outfile
    for dataset in datasets:
      format(src_dir, outfile, dataset)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
