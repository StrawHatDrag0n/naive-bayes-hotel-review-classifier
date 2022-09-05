import argparse
from file_utils import read_training_data
from nb_classifier import MultinomialNaiveBayesClassifier
from preprocess import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('input_folder_path', type=str,
                        help='input folder path')
    args = parser.parse_args()
    data, labels = read_training_data(args.input_folder_path)
    text_data = [d[0] for d in data]
    text_data = preprocess(text_data)
    model = MultinomialNaiveBayesClassifier(1)
    model.fit(text_data, labels)
    model.save_model()
