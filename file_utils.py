import argparse
import os
import glob
from typing import List, Tuple

NEGATIVE_DECEPTIVE_PATH = os.path.join('negative_polarity', 'deceptive_from_MTurk')
NEGATIVE_TRUTHFUL_PATH = os.path.join('negative_polarity', 'truthful_from_Web')
POSITIVE_DECEPTIVE_PATH = os.path.join('positive_polarity', 'deceptive_from_MTurk')
POSITIVE_TRUTHFUL_PATH = os.path.join('positive_polarity', 'truthful_from_TripAdvisor')

NEGATIVE_DECEPTIVE = 0
NEGATIVE_TRUTHFUL = 1
POSITIVE_DECEPTIVE = 2
POSITIVE_TRUTHFUL = 3

LABEL_A_LABEL_B_MAPPING = {
    NEGATIVE_DECEPTIVE: ('negative', 'deceptive'),
    NEGATIVE_TRUTHFUL: ('negative', 'truthful'),
    POSITIVE_DECEPTIVE: ('positive', 'deceptive'),
    POSITIVE_TRUTHFUL: ('positive', 'truthful'),
}

CLASS_LABELS = [
    (NEGATIVE_DECEPTIVE_PATH, NEGATIVE_DECEPTIVE),
    (NEGATIVE_TRUTHFUL_PATH, NEGATIVE_TRUTHFUL),
    (POSITIVE_DECEPTIVE_PATH, POSITIVE_DECEPTIVE),
    (POSITIVE_TRUTHFUL_PATH, POSITIVE_TRUTHFUL)
]
stop_words = {}


def read_training_class_data(class_data_path: str) -> List[Tuple[str, str]]:
    data: List[Tuple[str, str]] = list()
    folders = list(filter(lambda folder: os.path.isdir(os.path.join(class_data_path, folder)), os.listdir(class_data_path)))
    for folder in folders:
        if folder.startswith('.'): continue
        files = os.listdir(os.path.join(class_data_path, folder))
        for file_name in files:
            with open(os.path.join(class_data_path, folder, file_name)) as file:
                file_text = file.read()
                data.append((file_text, str(os.path.join(class_data_path, folder, file_name))))
    return data


def read_training_data_v1(data_path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    data: List[Tuple[str, str]] = list()
    labels: List[int] = list()

    for class_data_path, class_label in CLASS_LABELS:
        class_data = read_training_class_data(os.path.join(data_path, class_data_path))
        data.extend(class_data)
        labels.extend([class_label] * len(class_data))

    return data, labels


def read_training_data(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    return read_training_data_v2(path)


def read_data(paths):
    data = list()
    for path in paths:
        with open(path) as f:
            data.append((f.read(), path))
    return data


def read_training_data_v2(data_path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    data: List[Tuple[str, str]] = list()
    labels: List[int] = list()

    negative_truthful_paths = glob.glob(str(os.path.join(data_path, 'negative*', 'truthful*', '*', '*.txt')))
    negative_deceptive_paths = glob.glob(str(os.path.join(data_path, 'negative*', 'deceptive*', '*', '*.txt')))
    positive_truthful_paths = glob.glob(str(os.path.join(data_path, 'positive*', 'truthful*', '*', '*.txt')))
    positive_deceptive_paths = glob.glob(str(os.path.join(data_path, 'positive*', 'deceptive*', '*', '*.txt')))

    data.extend(read_data(negative_truthful_paths))
    labels.extend([NEGATIVE_TRUTHFUL]*len(negative_truthful_paths))

    data.extend(read_data(negative_deceptive_paths))
    labels.extend([NEGATIVE_DECEPTIVE] * len(negative_deceptive_paths))

    data.extend(read_data(positive_truthful_paths))
    labels.extend([POSITIVE_TRUTHFUL] * len(positive_truthful_paths))

    data.extend(read_data(positive_deceptive_paths))
    labels.extend([POSITIVE_DECEPTIVE] * len(positive_deceptive_paths))

    return data, labels


def read_testing_data_v1(data_path: str):
    path_list = list()
    queue = list()
    queue.append(data_path)
    data = list()
    while len(queue) > 0:
        path = queue.pop()
        file_and_folders = os.listdir(path)
        if any([os.path.isdir(os.path.join(path, f)) for f in file_and_folders]):
            for f in file_and_folders:
                if os.path.isdir(os.path.join(path, f)):
                    queue.append(os.path.join(path, f))
        else:
            for f in file_and_folders:
                path_list.append(os.path.join(os.path.join(path, f)))
    for file_path in path_list:
        with open(os.path.join(file_path)) as file:
            file_text = file.read()
            data.append((file_text, file_path))
    return data


def read_testing_data_v2(path: str):
    paths = glob.glob(str(os.path.join(path, '*', '*', '*', '*.txt')))
    return read_data(paths)


def read_testing_data(path: str):
    return read_testing_data_v1(path)


def write_prediction(labels, paths, file_name='nboutput.txt'):
    output: List[str] = []
    for idx, label in enumerate(labels):
        label_a, label_b = LABEL_A_LABEL_B_MAPPING.get(label)
        output.append(f'{label_b} {label_a} {paths[idx]}\n')

    with open(file_name, 'w+') as output_file:
        output_file.writelines(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('input_folder_path', type=str,
                        help='input folder path')
    args = parser.parse_args()
    # data, labels = read_training_data(args.input_folder_path)
    read_testing_data(args.input_folder_path)
