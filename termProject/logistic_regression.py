import os
import numpy as np
from collections import defaultdict, Counter
from math import ceil
from random import Random
from scipy.special import expit  # logistic (sigmoid) function
import nltk.sentiment.util
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.stem import PorterStemmer


class LogisticRegression:

    def __init__(self, n_features=0):
        self.class_dict = {'neg': 0, 'pos': 1}
        self.feature_dict = {}
        # self.feature_counter = Counter()
        # self.feature_counter_sorted = []
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1)  # weights (and bias)
        self.curr_class = int
        self.stop = set(stopwords.words("english"))
        self.lexicon = set(opinion_lexicon.words())
        self.ps = PorterStemmer()

    def update_feat_dict(self, data_set):
        feature_idx = self.n_features
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name), encoding="latin1") as f:
                    doc = f.read().split()
                    for token in doc:
                        if "train" in root:
                            if token in self.lexicon:
                                if token not in self.feature_dict:
                                    self.feature_dict[token] = feature_idx
                                    feature_idx += 1
                                    self.n_features += 1

    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        self.update_feat_dict(data_set)

        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name), encoding="latin1") as f:
                    if "DS_Store" not in name:
                        filenames.append(name)
                    for key, idx in self.class_dict.items():
                        if key in root:
                            if ".DS_Store" not in name:
                                classes[name] = idx
                    doc = f.read().split()
                    documents[name] = self.featurize(doc)

        return filenames, classes, documents

    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        for token in document:
            if token in self.feature_dict:
                vector[self.feature_dict[token]] = 1

        vector[-1] = 1
        return vector

    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        # initialize weights to 0s
        self.theta = np.zeros(self.n_features + 1)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]

                # create and fill in matrix x and vector y
                feature_matrix = np.zeros((len(minibatch), self.n_features + 1))

                gold_y_vec = np.zeros(len(minibatch), )

                for count, filename in enumerate(minibatch):
                    gold_y_vec[count] = classes[filename]

                    feature_matrix[count] = documents[filename]

                # compute y_hat
                y_hats = expit(np.dot(feature_matrix, self.theta))

                # update loss - Lce(y_hat, y) =
                loss = np.sum([(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) for y, y_hat in
                               zip(gold_y_vec, y_hats)])

                # compute gradient
                avg_gradient = (np.dot(feature_matrix.transpose(),
                                       np.subtract(y_hats, gold_y_vec)) / len(minibatch))

                # update weights (and bias)
                self.theta = np.subtract(self.theta, (avg_gradient * eta))

            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # get most likely class
            results[name]["correct"] = classes[name]

            if expit(np.dot((documents[name]), self.theta)) > 0.5:
                results[name]["predicted"] = 1
            else:
                results[name]["predicted"] = 0

        return results

    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))

        for doc in results:
            if results[doc]['predicted'] == 0 and results[doc]['correct'] == 0:
                confusion_matrix[(1, 1)] += 1
            elif results[doc]['predicted'] == 0 and results[doc]['correct'] == 1:
                confusion_matrix[(1, 0)] += 1
            elif results[doc]['predicted'] == 1 and results[doc]['correct'] == 0:
                confusion_matrix[(0, 1)] += 1
            elif results[doc]['predicted'] == 1 and results[doc]['correct'] == 1:
                confusion_matrix[(0, 0)] += 1

        tp = confusion_matrix[(0, 0)]
        tn = confusion_matrix[(1, 1)]
        fp = confusion_matrix[(0, 1)]
        fn = confusion_matrix[(1, 0)]

        if (tp + fp) == 0:
            pos_precision = 0
        else:
            pos_precision = tp / (tp + fp)

        if (tn + fn) == 0:
            neg_precision = 0
        else:
            neg_precision = tn / (tn + fn)

        if (tp + fn) == 0:
            pos_recall = 0
        else:
            pos_recall = tp / (tp + fn)

        if (tn + fp) == 0:
            neg_recall = 0
        else:
            neg_recall = tn / (tn + fp)

        if (pos_precision + pos_recall) == 0:
            pos_f1_score = 0
        else:
            pos_f1_score = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall)

        if (neg_precision + neg_recall) == 0:
            neg_f1_score = 0
        else:
            neg_f1_score = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall)

        print("")
        print("Overall Accuracy: ", np.trace(confusion_matrix) / np.sum(confusion_matrix))
        print("")
        print("Scores for positive classification: ")
        print("Precision: ", pos_precision)
        print("Recall: ", pos_recall)
        print("F1: ", pos_f1_score)
        print("")
        print("Scores for negative classification: ")
        print("Precision: ", neg_precision)
        print("Recall: ", neg_recall)
        print("F1: ", neg_f1_score)


if __name__ == '__main__':
    lr = LogisticRegression(n_features=0)
    lr.train('movie_reviews/train', batch_size=3, n_epochs=10, eta=0.1)
    results = lr.test('movie_reviews/dev')
    lr.evaluate(results)
