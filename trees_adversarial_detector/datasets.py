import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

ADULT_DATA_PATH = "data/adult-census-income/adult.csv"
VOICE_DATA_PATH = "data/voice/voice.csv"
SHUTTLE_DATA_PATH = "data/shuttle/shuttle.trn"
# DIABETES_DATA_PATH = "data/diabetestype/Diabetestype.csv"
BANKNOTE_DATA_PATH = 'data/banknote/data_banknote_authentication.txt'
CAR_EVAL_DATA_PATH = 'data/car-evaluation/car.data'
CMC_DATA_PATH = 'data/contraceptive/cmc.data'
MESSIDOR_DATA_PATH = 'data/messidor/messidor_features.arff'
PHISHING_DATA_PATH = 'data/phishing/PhishingData.arff'
ELECTRICITY_DATA_PATH = 'data/electricity/electricity-normalized.arff'  # https://www.openml.org/d/151
KC1_DATA_PATH = 'data/KC1/'  # https://www.openml.org/d/1067
BREAST_CANCER_DATA_PATH = 'data/breast-cancer-wisconsin/breast-cancer-wisconsin.data'


class DatasetUnkownException(Exception):
    pass


def load_dataset(dataset_name):
    if dataset_name == 'shuttle':
        return load_prepare_shuttle_dataset()

    if dataset_name == 'voice':
        return load_prepare_voice_dataset()

    if dataset_name == 'adult':
        return load_prepare_adult_dataset()

    if dataset_name == 'banknote':
        return load_prepare_banknote_dataset()

    if dataset_name == 'car_eval':
        return load_prepare_car_eval_dataset()

    if dataset_name == 'cmc':
        return load_prepare_contraceptive_dataset()

    if dataset_name == 'messidor':
        return load_prepare_messidor_dataset()

    if dataset_name == 'phishing':
        return load_prepare_phishing_dataset()

    if dataset_name == 'electricity':
        return load_prepare_electricity_dataset()

    if dataset_name == 'mnist':
        return load_prepare_mnist_dataset()

    if dataset_name == 'fmnist':
        return load_prepare_fashion_mnist_dataset()

    if dataset_name == 'breast_cancer':
        return load_prepare_breast_cancer()

    if dataset_name == "mnist_normalized":
        return load_prepare_mnist_normalized_dataset()

    # if dataset_name == 'diabetes':
    #     return load_prepare_diabetestype_dataset()

    raise DatasetUnkownException(f'there is no familiar dataset with the name {dataset_name}')


def load_prepare_breast_cancer():
    df = pd.read_csv(BREAST_CANCER_DATA_PATH,
                     names=['sample', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
                            'marginal_adhesion', 'single_epit_cell_size', 'bare_nuclei', 'bland_chromatin',
                            'normal_nucleoli', 'mitoses', 'class'])

    df.replace('?', 0, inplace=True)  # TODO: change this - maybe 0 is not optimal
    y = (df['class'].values / 2) - 1
    X = df.drop(columns=['class']).values.astype(float)

    return X, y


def load_prepare_shuttle_dataset():
    df = pd.read_csv(SHUTTLE_DATA_PATH, delimiter=' ',
                     names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'label'])

    y = df['label'].values
    X = df.drop(columns=['label']).values

    # encode the label to start from 0
    return X, LabelEncoder().fit_transform(y)


# def load_prepare_diabetestype_dataset():
#     df = pd.read_csv(DIABETES_DATA_PATH)
#
#     # Handle categorical features
#     cat_features = ['Type']
#
#     for feat in cat_features:
#         enc = LabelEncoder()
#         df[feat] = enc.fit_transform(df[feat])
#
#     # Extract features
#     X = df.drop(columns=['Class']).values
#     y = df['Class'].values
#
#     return X, y


def load_prepare_voice_dataset():
    df = pd.read_csv(VOICE_DATA_PATH)

    # Extract the labels
    y = (df['label'] == "female").values

    # Extract features
    X = df.drop(columns=['label']).values

    return X, y


def load_prepare_adult_dataset(handle_cat='LE'):
    df = pd.read_csv(ADULT_DATA_PATH)

    # change ? to NaN
    df.replace('?', np.nan, inplace=True)

    # Remove all rows with NaN values
    df.dropna(inplace=True)

    # Extract labels
    y = (df['income'] != "<=50K").values

    # Handle categorical features
    cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                    'native.country']

    if handle_cat == 'OHE':
        df = pd.get_dummies(df, columns=cat_features)
    else:
        for feat in cat_features:
            enc = LabelEncoder()
            df[feat] = enc.fit_transform(df[feat])

    # Extract features
    X = df.drop(columns=['income'])

    return X.reset_index(drop=True), y


def load_prepare_banknote_dataset():
    df = pd.read_csv(BANKNOTE_DATA_PATH, names=['variance', 'skew', 'curtosis', 'entropy', 'class'])

    y = df['class'].values
    X = df.drop(columns=['class']).values

    return X, y


def load_prepare_car_eval_dataset():
    df = pd.read_csv(CAR_EVAL_DATA_PATH, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    df['buying'] = df['buying'].map({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    df['maint'] = df['maint'].map({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    df['doors'] = df['doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
    df['persons'] = df['persons'].map({'2': 2, '4': 4, 'more': 6})
    df['lug_boot'] = df['lug_boot'].map({'small': 0, 'med': 1, 'big': 2})
    df['safety'] = df['safety'].map({'low': 0, 'med': 1, 'high': 2})
    df['class'] = df['class'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})

    y = df['class'].values
    X = df.drop(columns=['class']).values

    # encode the label to start from 0
    return X, LabelEncoder().fit_transform(y)


def load_prepare_contraceptive_dataset():
    df = pd.read_csv(CMC_DATA_PATH,
                     names=['wife_age', 'wife_ed', 'husband_ed', 'num_child', 'wife_rel', 'wife_work', 'husband_occ',
                            'sol_index', 'media', 'class'])

    y = df['class'].values
    X = df.drop(columns=['class']).values

    return X, y


def load_prepare_electricity_dataset():
    df = pd.DataFrame(arff.loadarff(ELECTRICITY_DATA_PATH)[0])

    df['class'] = (df['class'] == b"UP").astype(int)
    df['day'] = df['day'].map({b'1': 1, b'2': 2, b'3': 3, b'4': 4, b'5': 5, b'6': 6, b'7': 7})

    y = df['class'].values
    X = df.drop(columns=['class']).values

    return X, y


def load_prepare_messidor_dataset():
    df = pd.DataFrame(arff.loadarff(MESSIDOR_DATA_PATH)[0])

    df['Class'] = df['Class'].astype(int)

    y = df['Class'].values
    X = df.drop(columns=['Class']).values

    return X, y


def load_prepare_phishing_dataset():
    df = pd.DataFrame(arff.loadarff(PHISHING_DATA_PATH)[0])

    for col in df.columns:
        df[col] = df[col].astype(int)

    y = df['Result'].values
    X = df.drop(columns=['Result']).values

    return X, y


def load_prepare_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    X = np.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])

    return pd.DataFrame(X.reshape(-1, 784), columns=[str(i) for i in range(784)]), y


def load_prepare_mnist_normalized_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    X = np.vstack([x_train, x_test]) / 255
    y = np.hstack([y_train, y_test])
    X = pd.DataFrame(X.reshape(-1, 784), columns=[str(i) for i in range(784)])

    intersting_indexes = np.where((y == 0) | (y == 6))[0]
    new_labels = np.copy(y)
    new_labels[(y == 6)] = 1

    X = X.iloc[intersting_indexes].values
    y = new_labels[intersting_indexes]

    return X, y


# def load_prepare_cifar10_dataset():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_prepare_cifar10_datas()
#
#     X = np.vstack([x_train, x_test])
#     y = np.hstack([y_train, y_test])
#
#     return pd.DataFrame(X.reshape(-1, 784), columns=[str(i) for i in range(784)]), y


def load_prepare_fashion_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X = np.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])
    X = pd.DataFrame(X.reshape(-1, 784), columns=[str(i) for i in range(784)])

    intersting_indexes = np.where((y == 0) | (y == 6))[0]
    new_labels = np.copy(y)
    new_labels[(y == 6)] = 1

    X = X.iloc[intersting_indexes].values
    y = new_labels[intersting_indexes]

    return X, y
