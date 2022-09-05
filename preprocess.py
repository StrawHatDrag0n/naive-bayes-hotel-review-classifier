import argparse
from collections import defaultdict
import re
from file_utils import read_training_data


def get_frequent_words(data):
    data = tokenize(data)
    word_frequencies = defaultdict(int)
    for sentence in data:
        for word in sentence:
            word_frequencies[word] += 1

    word_frequencies_ordered = sorted([(word_frequency, word) for word, word_frequency in word_frequencies.items()],
                                      reverse=True)
    return word_frequencies_ordered


def transform_token(token):
    token = token.lower()
    return token


def tokenize(data):
    new_data = list()
    pattern = re.compile(r'\s|(\\|--|\.|\)|\(|!+|\\|\?|\,|\"|\'|\/)', flags=0)
    for sentence in data:
        tokens = filter(lambda token: token, pattern.split(sentence))
        tokens = list(map(lambda token: transform_token(token), tokens))
        new_data.append(tokens)
    return new_data


def remove_stop_words(data, stop_words):
    return list(filter(lambda token: token in stop_words, data))


def preprocess(data):
    return tokenize(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('input_folder_path', type=str,
                        help='input folder path')
    args = parser.parse_args()
    data, labels = read_training_data(args.input_folder_path)
    data = preprocess(data)
    print(data)
