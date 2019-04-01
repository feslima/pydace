from pathlib import Path
from numpy import genfromtxt

CSV_DIR = Path(__file__).parent


def get_training_data():
    input_build_file = 'input_build_data.csv'
    output_build_file = 'output_build_data.csv'

    return (genfromtxt(CSV_DIR / input_build_file, delimiter=',', skip_header=1),
            genfromtxt(CSV_DIR / output_build_file, delimiter=',', skip_header=1))


def get_validation_data():
    input_val_file = 'input_val_data.csv'
    output_val_file = 'output_val_data.csv'

    return (genfromtxt(CSV_DIR / input_val_file, delimiter=',', skip_header=1),
            genfromtxt(CSV_DIR / output_val_file, delimiter=',', skip_header=1))


def get_prediction_data():
    prediction_file = 'Yhat.csv'

    return genfromtxt(CSV_DIR / prediction_file, delimiter=',', skip_header=1)


if __name__ == '__main__':
    input_train, output_train = get_training_data()
    input_val, output_val = get_validation_data()
    matlab_prediction = get_prediction_data()
