import argparse

from file_utils import read_testing_data, write_prediction
from nb_classifier import MultinomialNaiveBayesClassifier
from preprocess import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('input_folder_path', type=str,
                        help='input folder path')
    args = parser.parse_args()
    data = read_testing_data(args.input_folder_path)
    text_data = [d[0] for d in data]
    text_data = preprocess(text_data)
    model = MultinomialNaiveBayesClassifier.load_model()
    labels = model.predict(text_data)

    write_prediction(labels, [d[1] for d in data])
