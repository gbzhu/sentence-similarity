import tensorflow as tf
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import csv
from collections import Counter
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from flask import Flask, globals, Response, request
import json


class Sentence:
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [
            t.lower() for t in nltk.word_tokenize(normalized_sentence)
        ]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]


def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies


def run_avg_benchmark(sentences1,
                      sentences2,
                      model=None,
                      use_stoplist=False,
                      doc_freqs=None):

    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]
    tokens1 = sentences1.tokens_without_stop if use_stoplist else sentences1.tokens
    tokens2 = sentences2.tokens_without_stop if use_stoplist else sentences2.tokens

    tokens1 = [token for token in tokens1 if token in model]
    tokens2 = [token for token in tokens2 if token in model]

    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0

    tokfreqs1 = Counter(tokens1)
    tokfreqs2 = Counter(tokens2)

    weights1 = [
        tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
        for token in tokfreqs1
    ] if doc_freqs else None
    weights2 = [
        tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
        for token in tokfreqs2
    ] if doc_freqs else None

    embedding1 = np.average([model[token] for token in tokfreqs1],
                            axis=0,
                            weights=weights1).reshape(1, -1)
    embedding2 = np.average([model[token] for token in tokfreqs2],
                            axis=0,
                            weights=weights2).reshape(1, -1)

    sim = cosine_similarity(embedding1, embedding2)[0][0]
    return sim


PATH_TO_WORD2VEC = './data/word2vec/GoogleNews-vectors-negative300.bin.gz'
WORD2VEC = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC,
                                                           binary=True)
PATH_TO_FREQUENCIES_FILE = './data/frequencies.tsv'
PATH_TO_DOC_FREQUENCIES_FILE = './data/doc_frequencies.tsv'

# in order to compute weighted averages of word embeddings later,
# we are going to load a file with word frequencies.
# These word frequencies have been collected from Wikipedia
# and saved in a tab-separated file.

# frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE)
DOC_FREQUENCIES = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
DOC_FREQUENCIES["NUM_DOCS"] = 1288431

STOP = set(nltk.corpus.stopwords.words("english"))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():
    req_data = json.loads(globals.request.data)
    ss = list(req_data.values())[0]
    s1 = Sentence(ss[0])
    s2 = Sentence(ss[1])
    sim = run_avg_benchmark(sentences1=s1,
                            sentences2=s2,
                            model=WORD2VEC,
                            use_stoplist=False,
                            doc_freqs=DOC_FREQUENCIES)
    return Response(str(sim), mimetype='text/plain')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
