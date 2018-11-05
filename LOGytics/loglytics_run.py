"""
Created on Sat Jul  7 21:24:08 2018

@author : Ruthvic
"""


from __future__ import print_function
import os
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import netron
from flask import Flask, render_template, request, redirect, url_for, flash  # For flask implementation
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['txt', 'jpg', 'png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "someDataString"
title = "LOGytics"
heading = "Welcome to our page"

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# modify=ObjectId()
def redirect_url():
    return request.args.get('next') or \
           request.referrer or \
           url_for('index')


@app.route("/")
@app.route("/index")
def index():
    text = open('static/summary/' + 'lstm_sentence_summary' + '.txt', 'r+')
    summary = text.read()
    return render_template('index.html', t=title, h=heading, summary = summary)


@app.route('/modal.html')
def modal():
    return render_template('modal.html')

@app.route('/withoutmodel')
def withoutmodel():
    # Write your log statements here
    return render_template('without_model.html')

@app.route('/withmodel')
def withmodel():
    # taking the log file which is generated in this step and sending to the trained model
    path = "C:\\Users\\ruthv\\PycharmProjects\\LOGytics\\data\\latest\\1.txt"
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
        text = text.split('\n')
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 10
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        # print(i,sentence)
        for t, char in enumerate(sentence):
            # print(t,char)
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    '''
    # uncomment these if you are training the model for the first time
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    '''
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    # --------------- load & Summary ----------

    modelPath = "C:\\Users\\ruthv\\PycharmProjects\\LOGytics\\model\\lstm_sentence.hdf5"
    model = load_model(modelPath)

    with open('static/summary/' + 'lstm_sentence_summary' + '.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


    # ---------  testing model ----------------

    path = "C:\\Users\\ruthv\\PycharmProjects\\LOGytics\\data\\latest\\test.txt"
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
        text = text.split('\n')

    maxlen = 10
    step = 3
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        # print(i,sentence)
        for t, char in enumerate(sentence):
            # print(t,char)
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    generated = ''
    sentence = text
    sentence = ''.join(str(e) + "\n" for e in sentence)
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    print('----- seed ends --------')

    # open a file to write the generated log here
    file1 = open("C:\\Users\\ruthv\\PycharmProjects\\New_Logytics\\LOGytics\\data\\test1.txt", "a")
    for i in range(400):
        x_pred = np.zeros((1, maxlen, 339))
        for t, char in enumerate(sentence.split('\n')):
            # print (t,char,char_indices[char])
            if t < 10:
                if char in char_indices:
                    x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=1.0)
        if next_index in indices_char:
            next_char = indices_char[next_index]
            next_char = next_char + " "
            generated += next_char
            sentence = sentence[1:] + next_char
            # print(next_char)
            file1.write(next_char)
            sys.stdout.write(next_char)
            sys.stdout.flush()
    print()
    file1.close()
    file1 = open("C:\\Users\\ruthv\\PycharmProjects\\New_Logytics\\LOGytics\\data\\test1.txt", "r")
    # read the generated text and check for the error
    content = str(file1.read())
    result = ""
    if content.find("error"):
        result = "Next step might give an error"
    return render_template('with_model.html', r=result)

@app.route("/modelview")
def modelview():
    netron.start(file="C:\\Users\\ruthv\\PycharmProjects\\LOGytics\\model\\lstm_sentence.hdf5")
    text = open('static/summary/' + 'lstm_sentence_summary' + '.txt', 'r+')
    summary = text.read()
    return render_template('index.html', t=title, h=heading, summary=summary)


# uploading a log file call this function and stores in the directory mentioned in UPLOAD_FOLDER
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
            flash('File saved')

    return redirect(url_for('index'))


if __name__ == "__main__":
    # set to false in production
    app.run(debug=True)
