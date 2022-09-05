TRUTHFUL = 'truthful'
POSITIVE = 'positive'
DECEPTIVE = 'deceptive'
NEGATIVE = 'negative'


def measure_performance(path='nboutput.txt'):
    predictions = list()
    with open(path) as f:
        predictions.extend(f.readlines())
    ttruthful = 0
    ftruthful = 0
    tdeceptive = 0
    fdeceptive = 0
    tpositive = 0
    fpositive = 0
    tnegative = 0
    fnegative = 0
    for prediction in predictions:
        label_a, label_b, _, location = prediction.split()
        if label_a == TRUTHFUL:
            if TRUTHFUL in location:
                ttruthful += 1
            else:
                ftruthful += 1
        if label_a == DECEPTIVE:
            if DECEPTIVE in location:
                tdeceptive += 1
            else:
                fdeceptive += 1
        if label_b == POSITIVE:
            if POSITIVE in location:
                tpositive += 1
            else:
                fpositive += 1
        if label_b == NEGATIVE:
            if NEGATIVE in location:
                tnegative += 1
            else:
                fnegative += 1
    print('\t Precision\tRecal\tF1')
    print(f'Truthful: {ttruthful/(ttruthful + ftruthful)}')
    print(f'Deceptive: {tdeceptive / (tdeceptive + fdeceptive)}')
    print(f'Positive: {tpositive / (tpositive + fpositive)}')
    print(f'Truthful: {tnegative / (tnegative + fnegative)}')


if __name__ == '__main__':
    measure_performance()