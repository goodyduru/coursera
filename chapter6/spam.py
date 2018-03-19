import re
import snowballstemmer
import numpy as np
import scipy.io as matio
from sklearn import svm

def get_vocab_list():
    with open('vocab.txt', 'r') as v:
        arr = v.readlines()
        dict = {}
        for k, v in enumerate(arr):
            words = v.split('\t')
            dict[int(words[0])] = words[1].replace('\n', '')
    return dict


def process_email(content):
    vocab_list = get_vocab_list()
    word_indices = []
    l = 0
    content = content.lower()
    #handle html
    content = re.sub("<[^<>]+>", " ", content)
    #handle numbers
    content = re.sub("\d+", "number", content)
    #handle url
    content = re.sub("(http|https)://[\w\.\/]*", "httpaddr", content)
    #handle email
    content = re.sub("[\w\._]+@[\w\.]+", "emailaddr", content)
    #handle dollar sign
    content = re.sub("[$]+", "dollar", content)
    #replace punctuations
    content = re.sub("[@$\/#\.\-:&\*\+=\[\]\?!\(\)\{\},\'\'\">_<;%]", " ", content)
    content = re.sub("[\\t\\n]", " ", content)
    content = re.sub("[ ]+", " ", content)
    stemmer = snowballstemmer.stemmer('english')
    while len(content) > 0:
        (first, sep, content) = content.partition(" ")
        first = re.sub("[^a-zA-Z0-9]", "", first)
        word = stemmer.stemWord(first.strip())
        if len(word) < 1:
            continue
        m = len(vocab_list)
        i = 1
        while i <= m:
            if vocab_list[i] == word:
                word_indices.append(i)
                break
            i += 1
        if  (l + len(word) + 1) > 78:
            print("\n")
            l = 0
        print(word, " ")
        l += (len(word) + 1)
    print("\n\n===================================\n")
    return word_indices


def email_features(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    m = len(word_indices)
    i = 0
    while i < m:
        x[word_indices[i]] = 1
        i += 1;
    return x

with open('emailSample1.txt', 'r') as file:
    email_content = file.read()
    process_content = process_email(email_content)
    print(process_content)
    print("\n\n")

    features = email_features(process_content)
    print("Number Of Features", len(features))
    print("Number of non-zero features", np.sum(features == 1))

    file = matio.loadmat("spamTrain.mat")
    X, y = file["X"], np.array(file["y"]).ravel()
    C = 0.1
    lin = svm.SVC(kernel='linear', C=C)
    clf = lin.fit(X, y)
    p = clf.predict(X)
    print("Test Accuracy was", np.mean(np.where((p == y), 1, 0)) * 100)

    file = matio.loadmat("spamTest.mat")
    Xtest, ytest = file["Xtest"], np.array(file["ytest"]).ravel()
    p = clf.predict(Xtest)
    print("Test Accuracy was", np.mean(np.where((p == ytest), 1, 0)) * 100)

    p = clf.predict(features.T)
    print(p)
