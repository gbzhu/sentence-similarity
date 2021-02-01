from gensim.models import KeyedVectors

import nltk

import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, globals, Response, request
import json
import os

app = Flask(__name__)
Model_Path = './data/word2vec/word2vec.bin'
Model = None


class Sentence:
    def __init__(self, s):
        self.raw = s
        normalized_sentence = s.replace("‘", "'").replace("’", "'")
        self.tokens = [
            t.lower() for t in nltk.word_tokenize(normalized_sentence)]


def run_avg(sentences1, sentences2, model=None):

    sim = 0

    sentences1 = Sentence(sentences1)
    sentences2 = Sentence(sentences2)

    tokens1 = sentences1.tokens
    tokens2 = sentences2.tokens

    tokens1 = [token for token in tokens1 if token in model]
    tokens2 = [token for token in tokens2 if token in model]

    if len(tokens1) == 0 or len(tokens2) == 0:
        return sim

    tokfreqs1 = Counter(tokens1)
    tokfreqs2 = Counter(tokens2)

    weights1 = None
    weights2 = None

    embedding1 = np.average(
        [model[token] for token in tokfreqs1], axis=0, weights=weights1
    ).reshape(1, -1)
    embedding2 = np.average(
        [model[token] for token in tokfreqs2], axis=0, weights=weights2
    ).reshape(1, -1)

    sim = cosine_similarity(embedding1, embedding2)[0][0]
    return sim


@app.route('/', methods=['POST', 'GET'])
def main():
    global Model
    if Model is None:
        Model = KeyedVectors.load(Model_Path, mmap='r')
        Model.syn0norm = Model.syn0
    req_data = json.loads(globals.request.data)
    ss = list(req_data.values())[0]
    sim = run_avg(sentences1=ss[0], sentences2=ss[1], model=Model)
    return Response(str(sim), mimetype='text/plain')


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=5000)
