from gensim.models import Word2Vec, KeyedVectors
from flask import Flask, globals, Response, request
import json
import os

app = Flask(__name__)
PATH_TO_MODEL = './data/word2vec/GoogleNews-vectors-gensim-normed.bin'
MODEL = None


@app.route('/', methods=['POST', 'GET'])
def main():
    global MODEL
    if MODEL is None:
        MODEL = KeyedVectors.load(PATH_TO_MODEL, mmap='r')
        MODEL.syn0norm = MODEL.syn0
    req_data = json.loads(globals.request.data)
    ss = list(req_data.values())[0]
    sim = MODEL.wmdistance(ss[0], ss[1])
    return Response(str(sim), mimetype='text/plain')


if __name__ == "__main__":
    # port = int(os.getenv("PORT", 8080))
    # app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=5000)
