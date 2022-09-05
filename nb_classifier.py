import argparse
import math
from collections import defaultdict
import json
from file_utils import read_training_data
from preprocess import preprocess


class MultinomialNaiveBayesClassifier(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

        self.total_n_docs = 0
        self.vocabulary = set()
        self.class_labels = set()
        self.class_priors = defaultdict(float)
        self.n_docs_per_class = defaultdict(int)
        self.class_to_docs = defaultdict(list)
        self.conditionals = defaultdict(lambda: defaultdict(float))

    def calculate_class_labels(self, y):
        self.class_labels = set(y)

    def calculate_vocabulary(self, sentences):
        for sentence in sentences:
            for token in sentence:
                self.vocabulary.add(token)

    def calculate_n_docs_per_class(self, sentences, labels):
        for sentence, label in zip(sentences, labels):
            self.n_docs_per_class[label] += 1
            self.total_n_docs += 1
            self.class_to_docs[label].append(sentence)

    def calculate_class_priors(self):
        for class_label, n_docs in self.n_docs_per_class.items():
            self.class_priors[class_label] = n_docs / self.total_n_docs

    def calculate_conditional_probabilities(self):
        class_to_token = defaultdict(lambda: defaultdict(int))
        class_to_total_tokens = defaultdict(int)

        for class_label, sentences in self.class_to_docs.items():
            for sentence in sentences:
                for token in sentence:
                    class_to_token[class_label][token] += 1
                    class_to_total_tokens[class_label] += 1

        for token in self.vocabulary:
            for class_label in self.class_labels:
                self.conditionals[token][class_label] = (class_to_token[class_label][token] + self.alpha) / \
                                                        (class_to_total_tokens[class_label] + self.alpha * len(self.vocabulary))

    def fit(self, sentences, labels):
        self.calculate_class_labels(labels)
        self.calculate_vocabulary(sentences)
        self.calculate_n_docs_per_class(sentences, labels)
        self.calculate_class_priors()
        self.calculate_conditional_probabilities()

    def predict(self, sentences):
        labels = list()
        for idx, sentence in enumerate(sentences):
            class_probs = {k: math.log(v) for k, v in self.class_priors.items()}
            for token in sentence:
                for class_label in self.class_labels:
                    class_probs[class_label] += math.log(self.conditionals.get(token, dict()).get(class_label, 1))
            labels.append(max(class_probs, key=class_probs.get))
        return labels

    def save_model(self, file_name='nbmodel.txt'):
        with open(file_name, 'w') as file:
            model = dict()
            model['vocabulary'] = list(self.vocabulary)
            model['class_labels'] = list(self.class_labels)
            model['class_priors'] = self.class_priors
            model['conditionals'] = self.conditionals
            model['alpha'] = self.alpha
            file.write(json.dumps(model))

    def set_params(self, params):
        self.conditionals = { k: {int(kk): vv for kk, vv in v.items()} for k, v in params.get('conditionals', dict()).items()}
        self.class_priors = {int(k): v for k, v in params.get('class_priors', dict()).items()}
        self.vocabulary = params.get('vocabulary')
        self.alpha = params.get('alpha')
        self.class_labels = params.get('class_labels')

    @classmethod
    def load_model(cls, model_path='nbmodel.txt'):
        with open(model_path) as model_file:
            model_params = json.loads(model_file.read())
        model = MultinomialNaiveBayesClassifier()
        model.set_params(model_params)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('input_folder_path', type=str,
                        help='input folder path')
    args = parser.parse_args()
    data, labels = read_training_data(args.input_folder_path)
    text_data = [d[0] for d in data]
    text_data = preprocess(text_data)

    model = MultinomialNaiveBayesClassifier(alpha=1)
    model.fit(text_data, labels)
    model.save_model()
