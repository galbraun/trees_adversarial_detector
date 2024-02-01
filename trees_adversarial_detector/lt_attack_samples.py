import json
import shutil

import joblib
import numpy as np
from python_on_whales import docker
from sklearn.datasets import load_svmlight_file
import datetime


def build_lt_attack_image():
    return docker.buildx.build(context_path=r'/mnt/c/Msc/Thesis/lt_attack', progress='plain', tags='lt_attack')


def generate_lt_attack_samples_for_dataset(workdir, attack_dir, norm_type, model_path, train_input_path,
                                           test_input_path, set_name, config_file_header, max_adv_points=1000):
    print("Loading data", datetime.datetime.now())
    all_data_splits = joblib.load(workdir / 'dataset_splitted.jblib')
    all_y = []
    num_features = None
    for key, (x, y) in all_data_splits.items():
        all_y.append(y)
        num_features = x.shape[1]
    n_classes = len(np.unique(np.hstack(all_y)))

    model_path_model = model_path.parent / model_path.name.replace('.json', '.model')

    print("Copy model", datetime.datetime.now())
    shutil.copy2(model_path, attack_dir / 'dumped_adv' / 'models' / model_path.name)
    shutil.copy2(model_path_model, attack_dir / 'dumped_adv' / 'models' / model_path_model.name)

    adv_config = {
        "num_threads": 20,
        "enable_early_return": True,
        "num_classes": n_classes,
        "feature_start": 0,
        "num_attack_per_point": 10,
        "norm_type": norm_type,
        "search_mode": "lt-attack",
        "model": f'dumped_adv/models/{model_path.name}',
        "num_features": num_features
    }

    print("Prepare train data", datetime.datetime.now())
    X_train, _ = load_svmlight_file(str(train_input_path), n_features=num_features)
    train_points_count = 0
    with open(attack_dir / 'dumped_adv' / 'data' / train_input_path.name, 'w') as target_file:
        with open(train_input_path, 'r') as source_file:
            target_file.write('\n'.join([line for line in source_file.read().splitlines() if not line.startswith('#')]))

    print("Preapre train config", datetime.datetime.now())
    adv_config['inputs'] = f'dumped_adv/data/{train_input_path.name}'
    adv_config['adv_training_path'] = f"dumped_adv/{config_file_header}_adv_samples_train"
    adv_config['num_point'] = X_train.shape[0] if X_train.shape[0] < max_adv_points else max_adv_points
    train_attack_config_fname = f'{config_file_header}_train_attack_config.json'

    print("Prepare train attack config", datetime.datetime.now())
    with open(attack_dir / 'configs' / train_attack_config_fname, 'w') as json_file:
        json.dump(adv_config, json_file)
    print("Generate train adv samples", datetime.datetime.now())
    train_adv_vectors = lt_attack_generate_samples(attack_dir, train_attack_config_fname,
                                                   f"{config_file_header}_adv_samples_train")
    train_adv_samples = genearte_adv_sample_from_change(X_train, train_adv_vectors, norm_type, X_train.shape[0])
    joblib.dump(train_adv_samples, workdir / f'train_{set_name}_results.jblib', compress=1)

    print("Prepare test data", datetime.datetime.now())
    X_test, _ = load_svmlight_file(str(test_input_path), n_features=num_features)
    with open(attack_dir / 'dumped_adv' / 'data' / test_input_path.name, 'w') as target_file:
        with open(test_input_path, 'r') as source_file:
            target_file.write('\n'.join([line for line in source_file.read().splitlines() if not line.startswith('#')]))

    print("Preapre test config", datetime.datetime.now())
    adv_config['inputs'] = f'dumped_adv/data/{test_input_path.name}'
    adv_config['adv_training_path'] = f"dumped_adv/{config_file_header}_adv_samples_test"
    adv_config['num_point'] = X_test.shape[0] if X_test.shape[0] < max_adv_points else max_adv_points

    print("Prepare test attack config", datetime.datetime.now())
    test_attack_config_fname = f'{config_file_header}_test_attack_config.json'
    with open(attack_dir / 'configs' / test_attack_config_fname, 'w') as json_file:
        json.dump(adv_config, json_file)
    print("Generate test adv samples", datetime.datetime.now())
    test_adv_vectors = lt_attack_generate_samples(attack_dir, test_attack_config_fname,
                                                  f"{config_file_header}_adv_samples_test")
    test_adv_samples = genearte_adv_sample_from_change(X_test, test_adv_vectors, norm_type, X_test.shape[0])
    joblib.dump(test_adv_samples, workdir / f'test_{set_name}_results.jblib', compress=1)


def genearte_adv_sample_from_change(X, adv_vectors, norm, num_points):
    adv_samples = {}
    for i in range(num_points):
        if i in adv_vectors.keys():
            new_vec = np.asarray(np.copy(X[i, :].todense()))
            for j in adv_vectors[i][norm].keys():
                new_vec[0, j] = adv_vectors[i][norm][j]
            adv_samples[i] = {'model_classify_correct': True, 'adv_succ': True, 'adv_vector': new_vec[0]}
        else:
            adv_samples[i] = {'model_classify_correct': False, 'adv_succ': False, 'adv_vector': None}

    return adv_samples

def lt_attack_generate_samples(attack_dir, config_file_name, result_fname):
    with docker.run('lt_attack:latest', detach=True, tty=True, volumes=[
        (str(attack_dir / 'dumped_adv'), '/tree-ensemble-attack-main/dumped_adv'), ((str(attack_dir / 'configs')),
                                                                                    '/tree-ensemble-attack-main/configs')]) as lt_container:
        lt_container.execute(['./lt_attack', f'configs/{config_file_name}'])

    adv_vectors = {}
    with open(attack_dir / 'dumped_adv' / result_fname, 'r') as fh:
        for row in fh.read().splitlines():
            if 'best' in row:
                split_row_to_parts = row.split(' ')
                idx = int(split_row_to_parts[0])
                if idx not in adv_vectors.keys():
                    adv_vectors[idx] = {}
                norm = int(split_row_to_parts[2])
                adv_vectors[idx][norm] = {}
                # orig_label = split_row_to_parts[3]
                for part in split_row_to_parts[4:]:
                    feat_change_split = part.split(':')
                    feat_idx, feat_new_val = int(feat_change_split[0]), float(feat_change_split[1])
                    adv_vectors[idx][norm][feat_idx] = feat_new_val
    return adv_vectors



for dir_n in relevant_dirs:
    workdir = pathlib.Path('/mnt/c/Msc/Thesis/new_datasets_lt_attacks') / dir_n
    attack_dir = pathlib.Path('/mnt/c/Msc/Thesis/lt_attack/tree-ensemble-attack-main')
    norm_type = -1 if str(dir_n).split('_')[-1] == 'inf' else 2
    model_path = workdir / 'trees_model.json'
    train_input_path = workdir / 'xgboost_data.svmlight'
    test_input_path = workdir / 'final_test_data.svmlight'
    set_name = 'adv'
    config_file_header = dir_n
    generate_lt_attack_samples_for_dataset(workdir, attack_dir, norm_type, model_path, train_input_path, test_input_path, set_name, config_file_header)

    workdir = pathlib.Path('/mnt/c/Msc/Thesis/new_datasets_lt_attacks') / dir_n
    attack_dir = pathlib.Path('/mnt/c/Msc/Thesis/lt_attack/tree-ensemble-attack-main')
    norm_type = -1 if str(dir_n).split('_')[-1] == 'inf' else 2
    model_path = workdir / 'trees_model.json'
    train_input_path = workdir / 'detector_adv_train_data.svmlight'
    test_input_path = workdir / 'detector_adv_test_data.svmlight'
    set_name = 'detector_adv'
    config_file_header = dir_n
    generate_lt_attack_samples_for_dataset(workdir, attack_dir, norm_type, model_path, train_input_path, test_input_path, set_name, config_file_header)

for dir_n in relevant_dirs:
    workdir = pathlib.Path('/mnt/c/Msc/Thesis/new_datasets_lt_attacks') / dir_n
    attack_dir = pathlib.Path('/mnt/c/Msc/Thesis/lt_attack/tree-ensemble-attack-main')
    norm_type = -1 if str(dir_n).split('_')[-1] == 'inf' else 2
    model_path = workdir / 'trees_model.json'
    train_input_path = workdir / 'detector_adv_train_data.svmlight'
    test_input_path = workdir / 'detector_adv_test_data.svmlight'
    set_name = 'detector_adv'
    config_file_header = dir_n
    generate_lt_attack_samples_for_dataset(workdir, attack_dir, norm_type, model_path, train_input_path, test_input_path, set_name, config_file_header)



def generate_non_lt_attack_samples_for_dataset(workir, search_mode, norm_type, model_path, train_input_path,
                                               test_input_path, set_name, config_file_header, max_adv_points=1000):
    print("Loading data", datetime.datetime.now())
    all_data_splits = joblib.load(workdir / 'dataset_splitted.jblib')
    all_y = []
    num_features = None
    for key, (x, y) in all_data_splits.items():
        all_y.append(y)
        num_features = x.shape[1]
    n_classes = len(np.unique(np.hstack(all_y)))

    adv_config = {
        "num_threads": 20,
        "enable_early_return": True,
        "num_classes": n_classes,
        "feature_start": 0,
        "num_attack_per_point": 10,
        "norm_type": norm_type,
        "search_mode": search_mode,
        "model": model_path,
        "num_features": num_features
    }

    print("Preapre train config", datetime.datetime.now())
    X_train, _ = load_svmlight_file(str(train_input_path), n_features=num_features)
    adv_config['inputs'] = train_input_path
    adv_config['num_point'] = X_train.shape[0] if X_train.shape[0] < max_adv_points else max_adv_points

    train_attack_config_fname = f'{config_file_header}_train_attack_config.json'

    print("Prepare train attack config", datetime.datetime.now())
    with open(workir / train_attack_config_fname, 'w') as json_file:
        json.dump(adv_config, json_file)
    print("Generate train adv samples", datetime.datetime.now())
    train_adv_samples, _ = generate_adv_samples(workir / train_attack_config_fname)
    joblib.dump(train_adv_samples, workdir / f'train_{set_name}_results.jblib', compress=1)

    print("Preapre test config", datetime.datetime.now())
    X_test, _ = load_svmlight_file(str(test_input_path), n_features=num_features)
    adv_config['inputs'] = test_input_path
    adv_config['num_point'] = X_test.shape[0] if X_test.shape[0] < max_adv_points else max_adv_points

    test_attack_config_fname = f'{config_file_header}_test_attack_config.json'

    print("Prepare test attack config", datetime.datetime.now())
    with open(workir / test_attack_config_fname, 'w') as json_file:
        json.dump(adv_config, json_file)
    print("Generate test adv samples", datetime.datetime.now())
    test_adv_samples, _ = generate_adv_samples(workir / test_attack_config_fname)
    joblib.dump(test_adv_samples, workdir / f'test_{set_name}_results.jblib', compress=1)