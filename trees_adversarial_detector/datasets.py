import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
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
KC1_DATA_PATH = 'data/KC1/kc1.arff'  # https://www.openml.org/d/1067
SPEECH_DATA_PATH = 'data/speech/phpznF975.arff'  # https://www.openml.org/d/40910
SPAMBASE_DATA_PATH = 'data/spambase/spambase.arff'  # https://www.openml.org/d/44095
WAVEFORM_DATA_PATH = 'data/waveform-5000/waveform-5000.arff'  # https://www.openml.org/d/979
WIND_DATA_PATH = 'data/wind/wind.arff'  # https://www.openml.org/d/847
BREAST_CANCER_DATA_PATH = 'data/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
DRY_BEAN_DATA_PATH = 'data/dry-bean/Dry_Bean_Dataset.arff'
ONLINE_SHOPPERS_DATA_PATH = 'data/online-shoppers/online_shoppers_intention.csv'
BANK_MARKETING_DATA_PATH = 'data/bank-marketing/bank-additional-full.csv'
# https://archive-beta.ics.uci.edu/ml/datasets/madelon

ROBUSTTREE_BREAST_CANCER_TRAIN = 'robusttree_datasets_splits/breast_cancer_scale0.train'
ROBUSTTREE_BREAST_CANCER_TEST = 'robusttree_datasets_splits/breast_cancer_scale0.test'

ROBUSTTREE_CODRNA_TRAIN = 'robusttree_datasets_splits/cod-rna_s0'
ROBUSTTREE_CODRNA_TEST = 'robusttree_datasets_splits/cod-rna_s.t0'

ROBUSTTREE_DIABETES_TRAIN = 'robusttree_datasets_splits/diabetes_scale0.train'
ROBUSTTREE_DIABETES_TEST = 'robusttree_datasets_splits/diabetes_scale0.test'

ROBUSTTREE_IJCNN1_TRAIN = 'robusttree_datasets_splits/ijcnn1s0'
ROBUSTTREE_IJCNN1_TEST = 'robusttree_datasets_splits/ijcnn1s0.t'

ROBUSTTREE_WEBSPAM_TRAIN = 'robusttree_datasets_splits/webspam_wc_normalized_unigram.svm0.train'
ROBUSTTREE_WEBSPAM_TEST = 'robusttree_datasets_splits/webspam_wc_normalized_unigram.svm0.test'

ROBUSTTREE_MNIST26_TRAIN = 'robusttree_datasets_splits/binary_mnist0'
ROBUSTTREE_MNIST26_TEST = 'robusttree_datasets_splits/binary_mnist0.t'

ROBUSTTREE_MNIST_TRAIN = 'robusttree_datasets_splits/ori_mnist.train0'
ROBUSTTREE_MNIST_TEST = 'robusttree_datasets_splits/ori_mnist.test0'

ROBUSTTREE_COVTYPE_TRAIN = 'robusttree_datasets_splits/covtype.scale01.train0'
ROBUSTTREE_COVTYPE_TEST = 'robusttree_datasets_splits/covtype.scale01.test0'

ROBUSTTREE_FASHION_TRAIN = 'robusttree_datasets_splits/fashion.train0'
ROBUSTTREE_FASHION_TEST = 'robusttree_datasets_splits/fashion.test0'

ROBUSTTREE_HIGGS_TRAIN = 'robusttree_datasets_splits/HIGGS_s.train0'
ROBUSTTREE_HIGGS_TEST = 'robusttree_datasets_splits/HIGGS_s.test0'

ROBUSTTREE_SENSORLESS_TRAIN = 'robusttree_datasets_splits/Sensorless.scale.tr0'
ROBUSTTREE_SENSORLESS_TEST = 'robusttree_datasets_splits/Sensorless.scale.val0'


class DatasetUnkownException(Exception):
    pass


def load_robusttree_dataset(dataset_name):
    if dataset_name == "breast_cancer":
        return load_robusttree_breast_cancer()

    if dataset_name == "codrna":
        return load_robusttree_codrna()

    if dataset_name == "diabetes":
        return load_robusttree_diabetes()

    if dataset_name == "ijcnn1":
        return load_robusttree_ijcnn1()

    if dataset_name == "webspam":
        return load_robusttree_webspam()

    if dataset_name == "mnist26":
        return load_robusttree_mnist26()

    if dataset_name == "mnist":
        return load_robusttree_mnist()

    if dataset_name == "covtype":
        return load_robusttree_covtype()

    if dataset_name == "fashion":
        return load_robusttree_fashion()

    if dataset_name == "higgs":
        return load_robusttree_higgs()

    if dataset_name == "sensorless":
        return load_robusttree_sensorless()

    raise DatasetUnkownException(f'there is no familiar dataset with the name {dataset_name}')


def load_robusttree_breast_cancer():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_BREAST_CANCER_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_BREAST_CANCER_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_codrna():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_CODRNA_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_CODRNA_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_diabetes():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_DIABETES_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_DIABETES_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_ijcnn1():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_IJCNN1_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_IJCNN1_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_webspam():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_WEBSPAM_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_WEBSPAM_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_mnist26():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_MNIST26_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_MNIST26_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_mnist():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_MNIST_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_MNIST_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_covtype():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_COVTYPE_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_COVTYPE_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_fashion():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_FASHION_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_FASHION_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_higgs():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_HIGGS_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_HIGGS_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_robusttree_sensorless():
    X_train, y_train = load_svmlight_file(ROBUSTTREE_SENSORLESS_TRAIN)
    X_test, y_test = load_svmlight_file(ROBUSTTREE_SENSORLESS_TEST)

    return np.array(X_train.todense()), np.array(X_test.todense()), y_train, y_test


def load_dataset(dataset_name):
    if dataset_name == "dry_bean":
        return load_prepare_dry_bean_dataset()

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

    if dataset_name == "online_shoppers":
        return load_prepare_online_shoppers_dataset()

    if dataset_name == "bank_marketing":
        return load_prepare_bank_marketing()

    if dataset_name == "kc1":
        return load_prepare_kc1()

    if dataset_name == "speech":
        return load_prepare_speech()

    if dataset_name == "spambase":
        return load_prepare_spambase()

    if dataset_name == "waveform":
        return load_prepare_waveform()

    if dataset_name == "wind":
        return load_prepare_wind()

    # if dataset_name == 'diabetes':
    #     return load_prepare_diabetestype_dataset()

    raise DatasetUnkownException(f'there is no familiar dataset with the name {dataset_name}')


def load_prepare_wind():
    df = pd.DataFrame(arff.loadarff(WIND_DATA_PATH)[0])

    df['binaryClass'] = df['binaryClass'].map({b'N': 0, b'P': 1})

    y = df['binaryClass'].values
    X = df.drop(columns=['binaryClass']).values

    return X, y


def load_prepare_waveform():
    df = pd.DataFrame(arff.loadarff(WAVEFORM_DATA_PATH)[0])

    df['binaryClass'] = df['binaryClass'].map({b'N': 0, b'P': 1})

    y = df['binaryClass'].values
    X = df.drop(columns=['binaryClass']).values

    return X, y


def load_prepare_spambase():
    df = pd.DataFrame(arff.loadarff(SPAMBASE_DATA_PATH)[0])

    df['class'] = df['class'].map({b'0': 0, b'1': 1})

    y = df['class'].values
    X = df.drop(columns=['class']).values

    return X, y


def load_prepare_speech():
    df = pd.DataFrame(arff.loadarff(SPEECH_DATA_PATH)[0])

    df['Target'] = df['Target'].map({b'Anomaly': 1, b'Normal': 0})

    y = df['Target'].values
    X = df.drop(columns=['Target']).values

    return X, y


def load_prepare_kc1():
    df = pd.DataFrame(arff.loadarff(KC1_DATA_PATH)[0])

    df['defects'] = df['defects'].map({b'true': 1, b'false': 0})

    y = df['defects'].values
    X = df.drop(columns=['defects']).values

    return X, y


def load_prepare_bank_marketing():
    # https://archive-beta.ics.uci.edu/ml/datasets/bank+marketing
    df = pd.read_csv(BANK_MARKETING_DATA_PATH, delimiter=';')
    df = df.replace('unknown', np.nan).dropna()
    df = pd.get_dummies(df, columns=['job', 'poutcome'])

    df['marital'] = df['marital'].map({'married': 0, 'single': 1, 'divorced': 3})
    df['education'] = df['education'].map({'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
                                           'high.school': 4, 'university.degree': 5, 'professional.course': 6})
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['contact'] = df['contact'].map({'telephone': 1, 'cellular': 0})
    df['month'] = df['month'].map({'may': 5, 'jul': 7, 'aug': 8, 'jun': 6, 'nov': 11, 'apr': 4, 'oct': 10, 'sep': 9,
                                   'mar': 3, 'dec': 12})
    df['day_of_week'] = df['day_of_week'].map({'thu': 5, 'mon': 2, 'wed': 4, 'tue': 3, 'fri': 6})
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    y = df['y'].values
    X = df.drop(columns=['y']).values

    return X, y


def load_prepare_online_shoppers_dataset():
    # https://archive-beta.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset
    df = pd.read_csv(ONLINE_SHOPPERS_DATA_PATH)

    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)
    df['Month'] = df['Month'].map({'May': 5, 'Nov': 11, 'Mar': 3, 'Dec': 12, 'Oct': 10, 'Sep': 9, 'Aug': 8,
                                   'Jul': 7, 'June': 6, 'Feb': 2})
    df = pd.get_dummies(df, columns=['VisitorType'])

    y = df['Revenue'].values
    X = df.drop(columns=['Revenue']).values

    return X, y


def load_prepare_dry_bean_dataset():
    # https://archive-beta.ics.uci.edu/ml/datasets/dry+bean+dataset
    df = pd.DataFrame(arff.loadarff(DRY_BEAN_DATA_PATH)[0])

    df['Class'] = df['Class'].map(
        {b'DERMASON': 0, b'SIRA': 1, b'SEKER': 2, b'HOROZ': 3, b'CALI': 4, b'BARBUNYA': 5, b'BOMBAY': 6})

    y = df['Class'].values
    X = df.drop(columns=['Class']).values

    return X, y


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

    df = df.replace({-1: 0, 0: 1, 1: 2})
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
