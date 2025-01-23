import argparse
import json
import pathlib

import holoviews as hv
import joblib
import pandas as pd
import numpy as np
import umap
import xgboost
from bokeh.io import save as bokeh_save
from bokeh.resources import INLINE as bokeh_resources_inline
from holoviews import opts
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
from deslib.util.faiss_knn_wrapper import FaissKNNClassifier


from trees_adversarial_detector.attack import generate_adv_samples
from trees_adversarial_detector.boosting_detector import generate_embedding_by_rounds, \
    calculate_a_priori_class_switching, generate_embedding_by_round_new_samples, train_rounds_classifiers
from trees_adversarial_detector.datasets import load_dataset, load_robusttree_dataset
from trees_adversarial_detector.embedding_proc import extract_embedding_dataset, train_embedding, \
    extract_new_samples_embedding_dataset, train_embedding_new_set
from trees_adversarial_detector.tree_models import train_tree_model

hv.extension('bokeh')
renderer = hv.renderer('bokeh')


class ModelUnkownException(Exception):
    pass


template_config = {
    'dataset': {
        'dataset_name': 'voice'
    },
    'output': {
        'output_path_parent': 'thesis_results'
    },
    'dataset_split': {
        'test_size': 0.2
    },
    'original_features_knn':
        {
            'k_low': 2,
            'k_high': 20,
        },
    'xgboost_model':
        {
            'model_type': 'XGBoost',
            'model_params': {
                'random_state': 84,
                'n_jobs': -1,
                'n_estimators': 40,
                'max_depth': 5,
                'verbosity': 3,
            }
        },
    'adv_samples':
        {
            "num_threads": 20,
            "enable_early_return": True,
            "num_classes": 2,
            "feature_start": 0,
            "num_point": 50,
            "num_attack_per_point": 5,
            "norm_type": 2,
            "search_mode": "signopt"
        },
    'detector_adv_samples':
        {
            "num_threads": 20,
            "enable_early_return": True,
            "num_classes": 2,
            "feature_start": 0,
            "num_point": 100,
            "num_attack_per_point": 5,
            "norm_type": 2,
            "search_mode": "signopt"
        },
    'embed_model':
        {
            'main_model_dataset_size': 300000000,
            'adv_repr_dataset_size': 30000000,
            'test_repr_dataset_size': 30000000,
            'detector_normal_train_repr_dataset_size': 10000000,
            'detector_normal_test_repr_dataset_size': 10000000,
            'detector_adv_train_repr_dataset_size': 10000000,
            'detector_adv_test_repr_dataset_size': 10000000,
            'epochs_detector_normal_train_model': 3,
            'epochs_detector_normal_test_model': 3,
            'epochs_detector_adv_train_model': 3,
            'epochs_detector_adv_test_model': 3,
            'epochs_main_model': 2,
            'epochs_adv_model': 3,
            'epochs_test_model': 3,
            'samples_embd_dim': 20,
            'nodes_embd_dim': 10,
        },
    'detector_config':
        {
            'type': 'simple'
        },
    'exec_purpose': 'adv'

}


def _load_dataset(config, experiment_output, load_from_disk):
    # Split dataset
    if (experiment_output / 'dataset_splitted.jblib').exists() and load_from_disk:
        dataset_splitted = joblib.load(experiment_output / 'dataset_splitted.jblib')
        n_classes = joblib.load(experiment_output / 'n_classes.jblib')
    else:
        # Load dataset
        X, y = load_dataset(config['dataset']['dataset_name'])
        if type(X) == pd.DataFrame:
            X = X.values
        n_classes = len(np.unique(y))
        
        dataset_splitted = dataset_spliter(X, y)
        joblib.dump(dataset_splitted, experiment_output / 'dataset_splitted.jblib', compress=1)
        joblib.dump(n_classes, experiment_output / 'n_classes.jblib')
        for dataset_name in dataset_splitted.keys():
            dump_svmlight_file(dataset_splitted[dataset_name][0], dataset_splitted[dataset_name][1],
                               experiment_output / f'{dataset_name}_data.svmlight',
                               comment=f"{dataset_name} data")

    return dataset_splitted, n_classes


def _load_robusttree_dataset(config, experiment_output, load_from_disk):
    X_train, X_test, y_train, y_test = load_robusttree_dataset(config['dataset']['dataset_name'])

    n_classes = len(np.unique(np.hstack([y_test, y_train])))

    # Split dataset
    if (experiment_output / 'dataset_splitted.jblib').exists() and load_from_disk:
        dataset_splitted = joblib.load(experiment_output / 'dataset_splitted.jblib')
    else:
        dataset_splitted = dataset_spliter(X_train, y_train, need_test=False, X_test=X_test, y_test=y_test)
        joblib.dump(dataset_splitted, experiment_output / 'dataset_splitted.jblib', compress=1)
        for dataset_name in dataset_splitted.keys():
            dump_svmlight_file(dataset_splitted[dataset_name][0], dataset_splitted[dataset_name][1],
                               experiment_output / f'{dataset_name}_data.svmlight',
                               comment=f"{dataset_name} data")

    return dataset_splitted, n_classes


def _original_repr_knn_performance(X_train, X_test, y_train, y_test, config, experiment_output):
    if (experiment_output / f'original_rep_knn_performance.html').exists():
        return

    orignal_knn_k_low = config['original_features_knn']['k_low']
    orignal_knn_k_high = config['original_features_knn']['k_high']
    original_knn_k_total = orignal_knn_k_high - orignal_knn_k_low
    original_acc = []
    original_recall = []
    original_precision = []

    if len(np.unique(np.hstack([y_test, y_train]))) > 2:
        multiclass = True
    else:
        multiclass = False

    for i in tqdm(range(orignal_knn_k_low, orignal_knn_k_high), total=original_knn_k_total,
                  desc="Checking KNN for original feature space for different Ks"):
        neigh = FaissKNNClassifier(n_neighbors=i)
        neigh.fit(X_train.astype('float32'), y_train.astype('float32'))
        test_preds = neigh.predict(X_test.astype('float32'))
        if multiclass:
            original_acc.append(accuracy_score(y_test, test_preds))
            original_recall.append(recall_score(y_test, test_preds, average='macro'))
            original_precision.append(precision_score(y_test, test_preds, average='macro'))
        else:
            original_acc.append(accuracy_score(y_test, test_preds))
            original_recall.append(recall_score(y_test, test_preds))
            original_precision.append(precision_score(y_test, test_preds))

    recall = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_recall), label='recall')
    precision = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_precision), label='precision')
    acc = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_acc), label='accuracy')
    all_curves = (recall * precision * acc).opts(opts.Curve(tools=['hover'], width=1000))
    all_curves = all_curves.opts(legend_position='right', title='Original Rep Performance')
    bokeh_plot = renderer.get_plot(all_curves).state
    bokeh_save(bokeh_plot, experiment_output / 'original_rep_knn_performance.html', resources=bokeh_resources_inline)


def _train_xgboost_model(X_train, X_test, y_train, y_test, config, experiment_output, load_from_disk):
    if (experiment_output / f'trees_model.json').exists() and load_from_disk:
        if config['xgboost_model']['model_type'] == "XGBoost":
            trees_model = xgboost.XGBClassifier(**config['xgboost_model']['model_params'])
        elif config['xgboost_model']['model_type'] == "RandomForest":
            trees_model = xgboost.XGBRFClassifier(**config['xgboost_model']['model_params'])
        else:
            raise ModelUnkownException(
                f'there is no familiar model with the name {config["xgboost_model"]["model_type"]}')

        trees_model.load_model(experiment_output / f'trees_model_sklearn_version.json')
    else:
        trees_model = train_tree_model(config['xgboost_model']['model_type'], X_train, y_train,
                                       config['xgboost_model']['model_params'])
        trees_model.save_model(experiment_output / f'trees_model_sklearn_version.json')
        trees_model.get_booster().dump_model(experiment_output / 'trees_model.json', dump_format='json')
        trees_model.get_booster().save_model(experiment_output / 'trees_model.model')

    if not (experiment_output / f'xgboost_model_roc_curves.html').exists():
        n_classes = len(np.unique(np.hstack([y_test, y_train])))
        if n_classes > 2:
            multiclass = True

            y_train_score = trees_model.predict_proba(X_train)
            y_test_score = trees_model.predict_proba(X_test)

            fpr_train = dict()
            fpr_test = dict()

            tpr_train = dict()
            tpr_test = dict()

            roc_auc_train = dict()
            roc_auc_test = dict()

            #for i in range(len(np.unique(y_test))):
            for i in range(n_classes):
                binarized_train_labels = np.zeros_like(y_train)
                binarized_test_labels = np.zeros_like(y_test)

                test_relevant_ids = np.where(y_test == i)[0]
                train_relevant_ids = np.where(y_train == i)[0]

                binarized_train_labels[train_relevant_ids] = 1
                binarized_test_labels[test_relevant_ids] = 1

                fpr_train[i], tpr_train[i], _ = roc_curve(binarized_train_labels, y_train_score[:, i])
                roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

                fpr_test[i], tpr_test[i], _ = roc_curve(binarized_test_labels, y_test_score[:, i])
                roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

            all_fpr_train = np.unique(np.concatenate([fpr_train[i] for i in range(n_classes)]))
            all_fpr_test = np.unique(np.concatenate([fpr_test[i] for i in range(n_classes)]))

            mean_tpr_train = np.zeros_like(all_fpr_train)
            for i in range(n_classes):
                mean_tpr_train += np.interp(all_fpr_train, fpr_train[i], tpr_train[i])
            mean_tpr_train /= n_classes

            mean_tpr_test = np.zeros_like(all_fpr_test)
            for i in range(n_classes):
                mean_tpr_test += np.interp(all_fpr_test, fpr_test[i], tpr_test[i])
            mean_tpr_test /= n_classes

            fpr_train["macro"] = all_fpr_train
            tpr_train["macro"] = mean_tpr_train
            roc_auc_train["macro"] = auc(fpr_train["macro"], tpr_train["macro"])

            fpr_test["macro"] = all_fpr_test
            tpr_test["macro"] = mean_tpr_test
            roc_auc_test["macro"] = auc(fpr_test["macro"], tpr_test["macro"])

            train_roc = hv.Curve(zip(fpr_train["macro"], tpr_train["macro"]),
                                 label=f'train_roc (AUC = {roc_auc_train["macro"]})')
            test_roc = hv.Curve(zip(fpr_test["macro"], tpr_test["macro"]),
                                label=f'test_roc (AUC = {roc_auc_test["macro"]})')

            per_label_roc_train = []
            per_label_roc_test = []
            for i in range(n_classes):
                per_label_roc_train.append(
                    hv.Curve(zip(fpr_train[i], tpr_train[i]), label=f'{int(i)} - AUC = {roc_auc_train[i]}'))
                per_label_roc_test.append(
                    hv.Curve(zip(fpr_test[i], tpr_test[i]), label=f'{int(i)} - AUC = {roc_auc_test[i]}'))

            final_label_roc_train = per_label_roc_train[0]
            final_label_roc_test = per_label_roc_test[0]

            for l_roc in per_label_roc_train[1:]:
                final_label_roc_train *= l_roc

            for l_roc in per_label_roc_test[1:]:
                final_label_roc_test *= l_roc

            all_curves = (train_roc * test_roc + final_label_roc_train + final_label_roc_test).opts(
                opts.Curve(tools=['hover'], width=1000, height=1000)).redim(
                x=hv.Dimension('x', range=(-0.1, 1.1)))

        else:
            multiclass = False

            train_proba = trees_model.predict_proba(X_train)[:, 1]
            test_proba = trees_model.predict_proba(X_test)[:, 1]

            fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_proba)
            fpr_test, tpr_test, thresh_test = roc_curve(y_test, test_proba)

            train_roc = hv.Curve(zip(fpr_train, tpr_train),
                                 label=f'train_roc (AUC = {roc_auc_score(y_train, train_proba)})')
            test_roc = hv.Curve(zip(fpr_test, tpr_test), label=f'test_roc (AUC = {roc_auc_score(y_test, test_proba)})')
            all_curves = (train_roc * test_roc).opts(opts.Curve(tools=['hover'], width=1000, height=1000)).redim(
                x=hv.Dimension('x', range=(-0.1, 1.1)))
            all_curves = all_curves.opts(legend_position='right', title='XGBoost model ROC curves')

        bokeh_plot = renderer.get_plot(all_curves).state
        bokeh_save(bokeh_plot, experiment_output / 'xgboost_model_roc_curves.html', resources=bokeh_resources_inline)

    return trees_model


def _reduce_data_to_2d(X):
    return umap.UMAP(verbose=True).fit_transform(X)


def _original_dataset_2d_rep(experiment_output, X_train, X_test, y_train, y_test):
    if (experiment_output / f'original_repr_2d_scatter.html').exists():
        return 

    if len(np.unique(np.hstack([y_test, y_train]))) > 2:
        multiclass = True
    else:
        multiclass = False

    if multiclass:
        all_data = np.vstack([X_train, X_test])
        all_labels = np.hstack([y_train, y_test])
        is_train = np.hstack([np.ones(X_train.shape[0]), np.zeros(X_test.shape[0])]).astype(bool)

        all_data_in_2d = _reduce_data_to_2d(all_data)

        train_scatter = hv.Scatter(all_data_in_2d[is_train, :], label='train set')
        test_scatter = hv.Scatter(all_data_in_2d[~is_train, :], label='test set')

        set_scatter = (train_scatter * test_scatter).opts(opts.Scatter(tools=['hover'], width=1000, height=500, size=3))
        set_scatter = set_scatter.opts(legend_position='right', title='Original Dataset 2d representation')

        label_scatters = []
        for i in np.unique(np.hstack([y_test, y_train])):
            label_scatters.append(hv.Scatter(all_data_in_2d[np.where(all_labels == i)[0], :], label=f'{int(i)} label'))

        final_label_scatter = label_scatters[0]
        for l_scatter in label_scatters[1:]:
            final_label_scatter *= l_scatter

        label_scatter = (final_label_scatter).opts(
            opts.Scatter(tools=['hover'], width=1000, height=500, size=3))
        label_scatter = label_scatter.opts(legend_position='right', title='Original Dataset 2d representation')

    else:
        all_data = np.vstack([X_train, X_test])
        all_labels = np.hstack([y_train, y_test]).astype(bool)
        is_train = np.hstack([np.ones(X_train.shape[0]), np.zeros(X_test.shape[0])]).astype(bool)

        all_data_in_2d = _reduce_data_to_2d(all_data)

        train_scatter = hv.Scatter(all_data_in_2d[is_train, :], label='train set')
        test_scatter = hv.Scatter(all_data_in_2d[~is_train, :], label='test set')

        negative_scatter = hv.Scatter(all_data_in_2d[~all_labels, :], label='negative set')
        postivie_scatter = hv.Scatter(all_data_in_2d[all_labels, :], label='positive set')

        # TODO: split train and test to negative and positive

        set_scatter = (train_scatter * test_scatter).opts(opts.Scatter(tools=['hover'], width=1000, height=500, size=3))
        set_scatter = set_scatter.opts(legend_position='right', title='Original Dataset 2d representation')

        label_scatter = (negative_scatter * postivie_scatter).opts(
            opts.Scatter(tools=['hover'], width=1000, height=500, size=3))
        label_scatter = label_scatter.opts(legend_position='right', title='Original Dataset 2d representation')

    all_scatters = set_scatter + label_scatter

    bokeh_plot = renderer.get_plot(all_scatters).state
    bokeh_save(bokeh_plot, experiment_output / 'original_repr_2d_scatter.html', resources=bokeh_resources_inline)


def _with_adv_dataset_2d_rep(experiment_output, X_train, X_test, Adv_train, Adv_test):
    if (experiment_output / f'adv_samples_original_repr_2d_scatter.html').exists():
        return

    all_data = np.vstack([X_train, X_test, Adv_train, Adv_test])
    indexes = np.hstack([np.zeros(X_train.shape[0]),
                         np.ones(X_test.shape[0]),
                         np.ones(Adv_train.shape[0]) * 2,
                         np.ones(Adv_test.shape[0]) * 3])

    all_data_in_2d = _reduce_data_to_2d(all_data)

    normal_train_scatter = hv.Scatter(all_data_in_2d[(indexes == 0).astype(bool), :],
                                      label='Normal train samples').opts(size=3)
    normal_test_scatter = hv.Scatter(all_data_in_2d[(indexes == 1).astype(bool), :],
                                     label='Normal test samples').opts(size=3)
    adv_train_scatter = hv.Scatter(all_data_in_2d[(indexes == 2).astype(bool), :], label='Adv train samples').opts(
        size=8)
    adv_test_scatter = hv.Scatter(all_data_in_2d[(indexes == 3).astype(bool), :], label='Adv test samples').opts(
        size=8)

    all_scatters = (normal_train_scatter * normal_test_scatter * adv_train_scatter * adv_test_scatter).opts(
        opts.Scatter(tools=['hover'], width=1000, height=500))
    all_scatters = all_scatters.opts(legend_position='right',
                                     title='Normal and Adv sample 2d representation based on original features')

    bokeh_plot = renderer.get_plot(all_scatters).state
    bokeh_save(bokeh_plot, experiment_output / 'adv_samples_original_repr_2d_scatter.html',
               resources=bokeh_resources_inline)


def _generate_adv_samples(config, experiment_output, num_features, num_classes, load_from_disk, train_data_filename, train_data_size,
                          test_data_filename, test_data_size, set_name='adv'):
    if (experiment_output / f'train_{set_name}_results.jblib').exists() and load_from_disk:
        train_adv_samples = joblib.load(experiment_output / f'train_{set_name}_results.jblib')
    else:
        adv_config = config[f'{set_name}_samples']
        adv_config['num_features'] = num_features
        adv_config['model'] = str(experiment_output / 'trees_model.json')
        adv_config['num_classes'] = num_classes

        adv_config['inputs'] = str(experiment_output / train_data_filename)
        adv_config['num_point'] = train_data_size
        with open(experiment_output / 'train_attack_config.json', 'w') as json_file:
            json.dump(adv_config, json_file)
        train_adv_samples, train_final_statistics = generate_adv_samples(experiment_output / 'train_attack_config.json')
        joblib.dump(train_adv_samples, experiment_output / f'train_{set_name}_results.jblib', compress=1)
        joblib.dump(train_final_statistics, experiment_output / f'train_{set_name}_statistics.jblib', compress=1)

    X_adv_train = []
    for sample in train_adv_samples:
        if train_adv_samples[sample]['model_classify_correct'] and train_adv_samples[sample].get('adv_succ', False):
            X_adv_train.append(train_adv_samples[sample]['adv_vector'])
    X_adv_stacked_train = np.vstack(X_adv_train)

    if (experiment_output / f'test_{set_name}_results.jblib').exists() and load_from_disk:
        test_adv_samples = joblib.load(experiment_output / f'test_{set_name}_results.jblib')
    else:
        adv_config = config[f'{set_name}_samples']
        adv_config['num_features'] = num_features
        adv_config['model'] = str(experiment_output / 'trees_model.json')
        adv_config['num_classes'] = num_classes

        adv_config['inputs'] = str(experiment_output / test_data_filename)
        adv_config['num_point'] = test_data_size
        with open(experiment_output / 'test_attack_config.json', 'w') as json_file:
            json.dump(adv_config, json_file)
        test_adv_samples, test_final_statistics = generate_adv_samples(experiment_output / 'test_attack_config.json')
        joblib.dump(test_adv_samples, experiment_output / f'test_{set_name}_results.jblib', compress=1)
        joblib.dump(test_final_statistics, experiment_output / f'test_{set_name}_statistics.jblib', compress=1)

    X_adv_test = []
    for sample in test_adv_samples:
        if test_adv_samples[sample]['model_classify_correct'] and test_adv_samples[sample].get('adv_succ', False):
            X_adv_test.append(test_adv_samples[sample]['adv_vector'])
    X_adv_stacked_test = np.vstack(X_adv_test)

    return X_adv_stacked_train, X_adv_stacked_test


def _extract_main_embedding_dataset(config, experiment_output, X, trees_model, load_from_disk, n_classes):
    if (experiment_output / 'embedding_dataset_X.jblib').exists() and load_from_disk:
        embedding_X = joblib.load(experiment_output / 'embedding_dataset_X.jblib')
        embedding_y = joblib.load(experiment_output / 'embedding_dataset_y.jblib')
        num_nodes = joblib.load(experiment_output / 'main_num_nodes.jblib')
    else:
        embedding_X, embedding_y, num_nodes = extract_embedding_dataset(X, trees_model,
                                                                        config['embed_model'][
                                                                            'main_model_dataset_size'], n_classes)
        joblib.dump(embedding_X, experiment_output / 'embedding_dataset_X.jblib', compress=1)
        joblib.dump(embedding_y, experiment_output / 'embedding_dataset_y.jblib', compress=1)
        joblib.dump(num_nodes, experiment_output / 'main_num_nodes.jblib', compress=1)

    return embedding_X, embedding_y, num_nodes


def _extract_main_embedding_model(config, experiment_output, embedding_X, embedding_y, num_nodes, num_samples,
                                  num_features,
                                  load_from_disk):
    if (experiment_output / 'embedding_model.h5').exists() and load_from_disk:
        embedding_model = keras.models.load_model(experiment_output / "embedding_model.h5")
        # history = joblib.load(experiment_output / 'embedding_model_history.jblib')
        summary = joblib.load(experiment_output / 'embedding_model_summary.jblib')
    else:
        history, embedding_model, summary = train_embedding(embedding_X, embedding_y, num_nodes, num_samples,
                                                            num_features,
                                                            epochs=config['embed_model']['epochs_main_model'],
                                                            samples_embd_dim=config['embed_model']['samples_embd_dim'],
                                                            node_embd_dim=config['embed_model']['nodes_embd_dim'])
        embedding_model.save(experiment_output / "embedding_model.h5")
        # joblib.dump(history, experiment_output / 'embedding_model_history.jblib')
        joblib.dump(summary, experiment_output / 'embedding_model_summary.jblib', compress=1)

    return embedding_model, summary


def _extract_extra_embeddings_dataset(config, experiment_output, X, X_extra, trees_model, load_from_disk, dataset_name):
    if (experiment_output / f'embedding_{dataset_name}_dataset_X.jblib').exists() and load_from_disk:
        embedding_X_adv = joblib.load(experiment_output / f'embedding_{dataset_name}_dataset_X.jblib')
        embedding_y_adv = joblib.load(experiment_output / f'embedding_{dataset_name}_dataset_y.jblib')
        num_nodes_adv = joblib.load(experiment_output / f'{dataset_name}_num_nodes.jblib')
    else:
        embedding_X_adv, embedding_y_adv, num_nodes_adv = extract_new_samples_embedding_dataset(X, X_extra, trees_model,
                                                                                                config['embed_model'][
                                                                                                    f'{dataset_name}_repr_dataset_size'])
        joblib.dump(embedding_X_adv, experiment_output / f'embedding_{dataset_name}_dataset_X.jblib', compress=1)
        joblib.dump(embedding_y_adv, experiment_output / f'embedding_{dataset_name}_dataset_y.jblib', compress=1)
        joblib.dump(num_nodes_adv, experiment_output / f'{dataset_name}_num_nodes.jblib', compress=1)

    return embedding_X_adv, embedding_y_adv, num_nodes_adv


def _extract_extra_samples_representations(config, experiment_output, embedding_model, embedding_X_adv, embedding_y_adv,
                                           num_nodes,
                                           num_samples, num_features, load_from_disk, dataset_name):
    if (experiment_output / f'embedding_{dataset_name}_model.h5').exists() and load_from_disk:
        embedding_model_adv = keras.models.load_model(experiment_output / f"embedding_{dataset_name}_model.h5")
        # history_adv = joblib.load(experiment_output / 'embedding_adv_model_history.jblib')
        summary_adv = joblib.load(experiment_output / f'embedding_{dataset_name}_model_summary.jblib')
    else:
        train_samples_embedding = embedding_model.layers[3].get_weights()[0]
        train_nodes_embedding = embedding_model.layers[4].get_weights()[0]
        hidden_1_weights = embedding_model.layers[7].get_weights()[0]
        hidden_1_biases = embedding_model.layers[7].get_weights()[1]
        hidden_2_weights = embedding_model.layers[8].get_weights()[0]
        hidden_2_biases = embedding_model.layers[8].get_weights()[1]

        history_adv, embedding_model_adv, summary_adv = train_embedding_new_set(embedding_X_adv, embedding_y_adv,
                                                                                num_nodes,
                                                                                num_samples, num_features,
                                                                                train_samples_embedding,
                                                                                train_nodes_embedding, hidden_1_weights,
                                                                                hidden_1_biases, hidden_2_weights,
                                                                                hidden_2_biases,
                                                                                epochs=config['embed_model'][
                                                                                    f'epochs_{dataset_name}_model'],
                                                                                samples_embd_dim=config['embed_model'][
                                                                                    'samples_embd_dim'],
                                                                                node_embd_dim=config['embed_model'][
                                                                                    'nodes_embd_dim'])
        embedding_model_adv.save(experiment_output / f"embedding_{dataset_name}_model.h5")

        # joblib.dump(history_adv, experiment_output / 'embedding_adv_model_history.jblib')
        joblib.dump(summary_adv, experiment_output / f'embedding_{dataset_name}_model_summary.jblib', compress=1)

    return embedding_model_adv, summary_adv


def _compare_representations(X_train, X_train_embed_model, X_test, X_test_embed_model, y_train, y_test, config,
                             experiment_output, n_classes):
    original_acc = []
    original_recall = []
    original_precision = []
    embeding_acc = []
    embeding_recall = []
    embeding_precision = []
    orignal_knn_k_low = config['original_features_knn']['k_low']
    orignal_knn_k_high = config['original_features_knn']['k_high']
    original_knn_k_total = orignal_knn_k_high - orignal_knn_k_low

    X_train_embed = X_train_embed_model.layers[3].get_weights()[0]
    X_test_embed = X_test_embed_model.layers[3].get_weights()[0]

    for i in tqdm(range(orignal_knn_k_low, orignal_knn_k_high), total=original_knn_k_total,
                  desc="Checking KNN for original vs embedding space feature space for different Ks"):
        neigh_embed = FaissKNNClassifier(n_neighbors=i)
        neigh_embed.fit(X_train_embed.astype('float32'), y_train.astype('float32'))
        embed_preds = neigh_embed.predict(X_test_embed.astype('float32'))

        neigh_orig = FaissKNNClassifier(n_neighbors=i)
        neigh_orig.fit(X_train.astype('float32'), y_train.astype('float32'))
        orig_preds = neigh_orig.predict(X_test.astype('float32'))

        if n_classes > 2:
            original_acc.append(accuracy_score(y_test, orig_preds))
            original_recall.append(recall_score(y_test, orig_preds, average='macro'))
            original_precision.append(precision_score(y_test, orig_preds, average='macro'))

            embeding_acc.append(accuracy_score(y_test, embed_preds))
            embeding_recall.append(recall_score(y_test, embed_preds, average='macro'))
            embeding_precision.append(precision_score(y_test, embed_preds, average='macro'))
        else:
            original_acc.append(accuracy_score(y_test, orig_preds))
            original_recall.append(recall_score(y_test, orig_preds))
            original_precision.append(precision_score(y_test, orig_preds))

            embeding_acc.append(accuracy_score(y_test, embed_preds))
            embeding_recall.append(recall_score(y_test, embed_preds))
            embeding_precision.append(precision_score(y_test, embed_preds))

    orig_recall = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_recall), label='orig recall')
    embed_recall = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), embeding_recall), label='embed recall')

    orig_precision = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_precision),
                              label='orig precision')
    embed_precision = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), embeding_precision),
                               label='embed precision')

    orig_acc = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), original_acc), label='orig accuracy')
    embed_acc = hv.Curve(zip(range(orignal_knn_k_low, orignal_knn_k_high), embeding_acc), label='embed accuracy')

    recall_curves = (orig_recall * embed_recall).opts(opts.Curve(tools=['hover'], width=1000))
    precision_curves = (orig_precision * embed_precision).opts(opts.Curve(tools=['hover'], width=1000))
    acc_curves = (orig_acc * embed_acc).opts(opts.Curve(tools=['hover'], width=1000))

    all_curves = (recall_curves + precision_curves + acc_curves)

    bokeh_plot = renderer.get_plot(all_curves).state
    bokeh_save(bokeh_plot, experiment_output / 'original_vs_embed_rep_knn_performance.html',
               resources=bokeh_resources_inline)


def _embedding_2d_represenetation(experiment_output, embedding_model, embedding_model_adv, embedding_model_test,
                                  y_train, y_test, n_classes):
    train_samples_embedding = embedding_model.layers[3].get_weights()[0]
    adv_samples_weights = embedding_model_adv.layers[3].get_weights()[0]
    test_samples_weights = embedding_model_test.layers[3].get_weights()[0]

    all_data = np.vstack([train_samples_embedding, test_samples_weights, adv_samples_weights])
    all_data_in_2d = _reduce_data_to_2d(all_data)

    if n_classes == 2:
        indexes = np.hstack([y_train, y_test, 2 * np.ones(adv_samples_weights.shape[0])])

        normal_negative_scatter = hv.Scatter(all_data_in_2d[(indexes == 0).astype(bool), :],
                                             label='Normal negative samples').opts(size=5)
        normal_positive_scatter = hv.Scatter(all_data_in_2d[(indexes == 1).astype(bool), :],
                                             label='Normal positive samples').opts(size=5)
        adv_samples = hv.Scatter(all_data_in_2d[(indexes == 2).astype(bool), :], label='Adv samples').opts(
            size=10, color='green')

        all_scatters = (normal_negative_scatter * normal_positive_scatter * adv_samples).opts(
            opts.Scatter(tools=['hover'], width=1000, height=500))
    else:
        indexes = np.hstack([y_train, y_test, -1 * np.ones(adv_samples_weights.shape[0])])

        adv_samples = hv.Scatter(all_data_in_2d[(indexes == -1).astype(bool), :], label='Adv samples').opts(
            size=10, color='green')

        label_scatters = []
        for i in np.unique(np.hstack([y_test, y_train])):
            label_scatters.append(hv.Scatter(all_data_in_2d[np.where(indexes == i)[0], :], label=f'{int(i)} label'))

        final_label_scatter = label_scatters[0]
        for l_scatter in label_scatters[1:]:
            final_label_scatter *= l_scatter

        final_label_scatter *= adv_samples

        all_scatters = (final_label_scatter).opts(
            opts.Scatter(tools=['hover'], width=1000, height=500, size=3))

    all_scatters = all_scatters.opts(legend_position='right',
                                     title='Normal and Adv sample 2d representation based on embedding')

    bokeh_plot = renderer.get_plot(all_scatters).state
    bokeh_save(bokeh_plot, experiment_output / 'embedding_2d_representations.html',
               resources=bokeh_resources_inline)


def full_experiment_cycle(config, load_from_disk=True):
    output_parent = pathlib.Path(config['output']['output_path_parent'])
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n {config} \n !!!!!!!!! \n")

    if config['exec_purpose'] == "ablation":
        config['embed_model']['samples_embd_dim'] = samples_embed_dim
        experiment_output = output_parent / f"{config['dataset']['dataset_name']}_{config['adv_samples']['search_mode']}_{config['adv_samples']['norm_type']}_{config['embed_model']['samples_embd_dim']}_{config['embed_model']['nodes_embd_dim']}"
    else:
        experiment_output = output_parent / f"{config['dataset']['dataset_name']}_{config['adv_samples']['search_mode']}_{config['adv_samples']['norm_type']}"

    if not experiment_output.exists():
        experiment_output.mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset")
    if config['dataset']['dataset_name'] in ['breast_cancer', 'codrna', 'diabetes', 'ijcnn1', 'webspam', 'mnist26', 'mnist', 'covtype', 'fashion', 'higgs', 'sensorless']:
        dataset_splitted, n_classes = _load_robusttree_dataset(config, experiment_output, load_from_disk)
    else:
        dataset_splitted, n_classes = _load_dataset(config, experiment_output, load_from_disk)

    # Extract original representation KNN performance
    print("Extracting original representation KNN performance")
    _original_repr_knn_performance(dataset_splitted['knn_org_perf'][0], dataset_splitted['final_test'][0],
                                   dataset_splitted['knn_org_perf'][1], dataset_splitted['final_test'][1], config,
                                   experiment_output)

    # Train XGBoost Classifier
    print("Preparing XGBoost model")
    trees_model = _train_xgboost_model(dataset_splitted['xgboost'][0], dataset_splitted['final_test'][0],
                                       dataset_splitted['xgboost'][1], dataset_splitted['final_test'][1], config,
                                       experiment_output, load_from_disk)

    # Extract original dataset 2d representations
    print("Extracting 2D representation of original data")
    _original_dataset_2d_rep(experiment_output, dataset_splitted['xgboost'][0], dataset_splitted['final_test'][0],
                             dataset_splitted['xgboost'][1], dataset_splitted['final_test'][1])

    # Generate Adversarial Samples
    print("Preparing adversarial samples")
    X_adv_train, X_adv_test = _generate_adv_samples(config, experiment_output, dataset_splitted['xgboost'][0].shape[1],
                                                    n_classes,
                                                    load_from_disk, 'xgboost_data.svmlight', dataset_splitted['xgboost'][0].shape[0],
                                                    'final_test_data.svmlight', dataset_splitted['final_test'][0].shape[0])


    # Prepare data for Adv detector
    print("Preparing adversarial samples")
    X_detector_adv_train, X_detector_adv_test = _generate_adv_samples(config, experiment_output,
                                                                      dataset_splitted['detector_adv_train'][0].shape[
                                                                          1], n_classes,
                                                                      load_from_disk,
                                                                      'detector_adv_train_data.svmlight',
                                                                      dataset_splitted['detector_adv_train'][0].shape[0],
                                                                      'detector_adv_test_data.svmlight',
                                                                      dataset_splitted['detector_adv_test'][0].shape[0],
                                                                      'detector_adv')

    if config['exec_purpose'] == "adv":
        return


    # Extract adversarial dataset representation using original representation
    print("Extracting 2D representation with adversarial samples")
    _with_adv_dataset_2d_rep(experiment_output, dataset_splitted['xgboost'][0], dataset_splitted['final_test'][0],
                             X_adv_train, X_adv_test)

    # Extract embedding dataset
    print("Extracting dataset for embedding calculation")
    embedding_X, embedding_y, num_nodes = _extract_main_embedding_dataset(config, experiment_output,
                                                                          dataset_splitted['embedding_model'][0],
                                                                          trees_model, load_from_disk, n_classes)

    # Train Embedding model
    print("Training embedding representation for original dataset")
    embedding_model, summary = _extract_main_embedding_model(config, experiment_output, embedding_X,
                                                             embedding_y, num_nodes,
                                                             dataset_splitted['embedding_model'][0].shape[0],
                                                             dataset_splitted['embedding_model'][0].shape[1],
                                                             load_from_disk)

    # Extract adv embedding dataset
    embedding_X_adv, embedding_y_adv, num_nodes_adv = _extract_extra_embeddings_dataset(config, experiment_output,
                                                                                        dataset_splitted[
                                                                                            'embedding_model'][0],
                                                                                        np.vstack(
                                                                                            [X_adv_train, X_adv_test]),
                                                                                        trees_model, load_from_disk,
                                                                                        'adv')

    # Train adversarial Embedding representations
    print("Training embedding representation adv dataset")
    embedding_model_adv, summary_adv = _extract_extra_samples_representations(config, experiment_output,
                                                                              embedding_model,
                                                                              embedding_X_adv,
                                                                              embedding_y_adv, num_nodes_adv,
                                                                              X_adv_train.shape[0],
                                                                              X_adv_train.shape[1],
                                                                              load_from_disk, 'adv')

    # Extract test embedding dataset
    embedding_X_test, embedding_y_test, num_nodes_test = _extract_extra_embeddings_dataset(config, experiment_output,
                                                                                           dataset_splitted[
                                                                                               'embedding_model'][0],
                                                                                           dataset_splitted[
                                                                                               'final_test'][0],
                                                                                           trees_model, load_from_disk,
                                                                                           'test')

    # Train test Embedding representations
    print("Training embedding representation test dataset")
    embedding_model_test, summary_test = _extract_extra_samples_representations(config, experiment_output,
                                                                                embedding_model,
                                                                                embedding_X_test,
                                                                                embedding_y_test, num_nodes_test,
                                                                                dataset_splitted['final_test'][0].shape[
                                                                                    0],
                                                                                dataset_splitted['final_test'][0].shape[
                                                                                    1],
                                                                                load_from_disk, 'test')

    # Extract embedding 2d representation
    print("Extract 2d representation of the embeddings")
    _embedding_2d_represenetation(experiment_output, embedding_model, embedding_model_adv, embedding_model_test,
                                  dataset_splitted['embedding_model'][1], dataset_splitted['final_test'][1], n_classes)

    # Extract representation comparison
    _compare_representations(dataset_splitted['embedding_model'][0], embedding_model, dataset_splitted['final_test'][0],
                             embedding_model_test, dataset_splitted['embedding_model'][1],
                             dataset_splitted['final_test'][1], config, experiment_output, n_classes)


    if config['detector_config']['type'] == "simple":
        # Extract detector normal train embedding dataset
        embedding_X_detector_normal_train, embedding_y_detector_normal_train, num_nodes_detector_normal_train = \
            _extract_extra_embeddings_dataset(config, experiment_output, dataset_splitted['embedding_model'][0],
                                              dataset_splitted[
                                                  'detector_normal_train'][0],
                                              trees_model, load_from_disk,
                                              'detector_normal_train')

        # Train normal train Embedding representations
        print("Training embedding representation detector_normal_train dataset")
        embedding_model_detector_normal_train, summary_detector_normal_train = _extract_extra_samples_representations(
            config,
            experiment_output,
            embedding_model,
            embedding_X_detector_normal_train,
            embedding_y_detector_normal_train,
            num_nodes_detector_normal_train,
            dataset_splitted['detector_normal_train'][0].shape[
                0],
            dataset_splitted['detector_normal_train'][0].shape[
                1],
            load_from_disk, 'detector_normal_train')

        # Extract detector normal test embedding dataset
        embedding_X_detector_normal_test, embedding_y_detector_normal_test, num_nodes_detector_normal_test = \
            _extract_extra_embeddings_dataset(config, experiment_output, dataset_splitted['embedding_model'][0],
                                              dataset_splitted[
                                                  'detector_normal_test'][0],
                                              trees_model, load_from_disk,
                                              'detector_normal_test')

        # Train normal test Embedding representations
        print("Training embedding representation detector_normal_test dataset")
        embedding_model_detector_normal_test, summary_detector_normal_test = _extract_extra_samples_representations(
            config,
            experiment_output,
            embedding_model,
            embedding_X_detector_normal_test,
            embedding_y_detector_normal_test,
            num_nodes_detector_normal_test,
            dataset_splitted[
                'detector_normal_test'][
                0].shape[
                0],
            dataset_splitted[
                'detector_normal_test'][
                0].shape[
                1],
            load_from_disk,
            'detector_normal_test')

        # Extract detector adv train embedding dataset
        embedding_X_detector_adv_train, embedding_y_detector_adv_train, num_nodes_detector_adv_train = \
            _extract_extra_embeddings_dataset(config, experiment_output, dataset_splitted['embedding_model'][0],
                                              X_detector_adv_train,
                                              trees_model, load_from_disk,
                                              'detector_adv_train')

        # Train normal train Embedding representations
        print("Training embedding representation detector_normal_train dataset")
        embedding_model_detector_adv_train, summary_detector_adv_train = _extract_extra_samples_representations(config,
                                                                                                                experiment_output,
                                                                                                                embedding_model,
                                                                                                                embedding_X_detector_adv_train,
                                                                                                                embedding_y_detector_adv_train,
                                                                                                                num_nodes_detector_adv_train,
                                                                                                                len(
                                                                                                                    X_detector_adv_train),
                                                                                                                dataset_splitted[
                                                                                                                    'detector_adv_train'][
                                                                                                                    0].shape[
                                                                                                                    1],
                                                                                                                load_from_disk,
                                                                                                                'detector_adv_train')

        # Extract detector adv train embedding dataset
        embedding_X_detector_adv_test, embedding_y_detector_adv_test, num_nodes_detector_adv_test = \
            _extract_extra_embeddings_dataset(config, experiment_output, dataset_splitted['embedding_model'][0],
                                              X_detector_adv_test,
                                              trees_model, load_from_disk,
                                              'detector_adv_test')

        # Train normal train Embedding representations
        print("Training embedding representation detector_normal_test dataset")
        embedding_model_detector_adv_test, summary_detector_adv_test = _extract_extra_samples_representations(config,
                                                                                                              experiment_output,
                                                                                                              embedding_model,
                                                                                                              embedding_X_detector_adv_test,
                                                                                                              embedding_y_detector_adv_test,
                                                                                                              num_nodes_detector_adv_test,
                                                                                                              len(
                                                                                                                  X_detector_adv_test),
                                                                                                              dataset_splitted[
                                                                                                                  'detector_adv_test'][
                                                                                                                  0].shape[
                                                                                                                  1],
                                                                                                              load_from_disk,
                                                                                                              'detector_adv_test')

        _check_simple_detector(dataset_splitted['detector_normal_train'][0],
                               dataset_splitted['detector_normal_test'][0],
                               X_detector_adv_train, X_detector_adv_test,
                               embedding_model_detector_normal_train, embedding_model_detector_normal_test,
                               embedding_model_detector_adv_train,
                               embedding_model_detector_adv_test, experiment_output)
    else:
        _check_boosting_detector(config, trees_model, dataset_splitted['embedding_model'][0], dataset_splitted[
            'detector_normal_train'][0], dataset_splitted['detector_normal_train'][1], dataset_splitted[
                                     'detector_normal_test'][0], X_detector_adv_train, X_detector_adv_test
                                 , experiment_output, load_from_disk, n_classes)

    # Evaluate Adv detector
    # check_simple_detector(config['dataset']['dataset_name'], experiment_output / config['dataset']['dataset_name'],
    #                       X.shape[1])

    # Boosting Detector
    pass


def _check_boosting_detector(config, trees_model, X_basic_embedding_model, X_original_normal_train,
                             y_original_normal_train,
                             X_original_normal_test, X_original_adv_train, X_original_adv_test,
                             experiment_output, load_from_disk, n_classes):
    samples_embd_dim = config['embed_model']['samples_embd_dim']
    node_embd_dim = config['embed_model']['nodes_embd_dim']

    if (experiment_output / 'boosting_detector_base_models_0.h5').exists() and load_from_disk:
        histories = []
        embedding_models = []
        summaries = []
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            # histories.append(joblib.load(experiment_output / f'boosting_detector_base_histories_{i}.jblib'))
            embedding_models.append(
                keras.models.load_model(experiment_output / f'boosting_detector_base_models_{i}.h5'))
            # summaries.append(joblib.load(experiment_output / f'boosting_detector_base_summaries_{i}.jblib'))
    else:
        histories, embedding_models, summaries = generate_embedding_by_rounds(X_basic_embedding_model, trees_model,
                                                                              12000000, n_classes,
                                                                              samples_embd_dim=samples_embd_dim,
                                                                              node_embd_dim=node_embd_dim)
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            embedding_models[i].save(experiment_output / f"boosting_detector_base_models_{i}.h5")
            # joblib.dump(histories[i], experiment_output / 'boosting_detector_base_histories_[i].jblib')
            # joblib.dump(summaries[i], experiment_output / f'boosting_detector_base_summaries_{i}.jblib', compress=1)

    # X_original_normal_and_adv_train = np.vstack([X_original_normal_train, X_original_adv_train])
    # y_original_noraml_and_adv_train = np.hstack(
    #     [np.zeros(len(X_original_normal_train)), np.ones(len(X_original_adv_train))])

    if (experiment_output / 'boosting_detector_train_normal_models_0.h5').exists() and load_from_disk:
        train_normal_histories = []
        train_normal_embedding_models = []
        train_normal_summaries = []
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            # train_normal_histories.append(joblib.load(experiment_output / f'boosting_detector_train_normal_histories_{i}.jblib'))
            train_normal_embedding_models.append(
                keras.models.load_model(experiment_output / f'boosting_detector_train_normal_models_{i}.h5'))
            # train_normal_summaries.append(joblib.load(experiment_output / f'boosting_detector_train_normal_summaries_{i}.jblib'))
    else:
        train_normal_histories, train_normal_embedding_models, train_normal_summaries = generate_embedding_by_round_new_samples(
            X_basic_embedding_model,
            X_original_normal_train,
            trees_model,
            5000000,
            embedding_models,
            n_classes,
            samples_embd_dim=samples_embd_dim,
            node_embd_dim=node_embd_dim
        )
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            train_normal_embedding_models[i].save(experiment_output / f"boosting_detector_train_normal_models_{i}.h5")
            # joblib.dump(train_normal_histories[i], experiment_output / 'boosting_detector_train_normal_histories_[i].jblib')
            # joblib.dump(train_normal_summaries[i], experiment_output / f'boosting_detector_train_normal_summaries_{i}.jblib', compress=1)

    # if (experiment_output / 'boosting_detector_round_classifiers.jblib').exists() and load_from_disk:
    #     round_classifiers = joblib.load('boosting_detector_round_classifiers.jblib')
    # else:
    round_classifiers = train_rounds_classifiers(train_normal_embedding_models, y_original_normal_train,
                                                 trees_model.n_estimators)
    # joblib.dump(round_classifiers, experiment_output / 'boosting_detector_round_classifiers.jblib')

    train_normal_predictions = []
    for i in range(trees_model.n_estimators):
        pca_projector = round_classifiers[i][1]
        knn_classifier = round_classifiers[i][0]

        train_samples_embedding = pca_projector.transform(train_normal_embedding_models[i].layers[3].get_weights()[0])

        train_normal_predictions.append(knn_classifier.predict(train_samples_embedding.astype('float32')))

    train_normal_probabilities = calculate_a_priori_class_switching(np.array(train_normal_predictions).T)

    if (experiment_output / 'boosting_detector_train_adv_models_0.h5').exists() and load_from_disk:
        train_adv_histories = []
        train_adv_embedding_models = []
        train_adv_summaries = []
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            # train_adv_histories.append(joblib.load(experiment_output / f'boosting_detector_train_adv_histories_{i}.jblib'))
            train_adv_embedding_models.append(
                keras.models.load_model(experiment_output / f'boosting_detector_train_adv_models_{i}.h5'))
            # train_adv_summaries.append(joblib.load(experiment_output / f'boosting_detector_train_adv_summaries_{i}.jblib'))
    else:
        train_adv_histories, train_adv_embedding_models, train_adv_summaries = generate_embedding_by_round_new_samples(
            X_basic_embedding_model,
            X_original_adv_train,
            trees_model,
            5000000,
            embedding_models,
            n_classes,
            samples_embd_dim=samples_embd_dim,
            node_embd_dim=node_embd_dim
        )
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            train_adv_embedding_models[i].save(experiment_output / f"boosting_detector_train_adv_models_{i}.h5")
            # joblib.dump(train_adv_histories[i], experiment_output / 'boosting_detector_train_adv_histories_[i].jblib')
            # joblib.dump(train_adv_summaries[i], experiment_output / f'boosting_detector_train_adv_summaries_{i}.jblib', compress=1)

    train_adv_predictions = []
    for i in range(trees_model.n_estimators):
        pca_projector = round_classifiers[i][1]
        knn_classifier = round_classifiers[i][0]

        train_samples_embedding = pca_projector.transform(train_adv_embedding_models[i].layers[3].get_weights()[0])

        train_adv_predictions.append(knn_classifier.predict(train_samples_embedding.astype('float32')))

    if (experiment_output / 'boosting_detector_test_normal_models_0.h5').exists() and load_from_disk:
        test_normal_histories = []
        test_normal_embedding_models = []
        test_normal_summaries = []
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            # test_normal_histories.append(joblib.load(experiment_output / f'boosting_detector_test_normal_histories_{i}.jblib'))
            test_normal_embedding_models.append(
                keras.models.load_model(experiment_output / f'boosting_detector_test_normal_models_{i}.h5'))
            # test_normal_summaries.append(joblib.load(experiment_output / f'boosting_detector_test_normal_summaries_{i}.jblib'))
    else:
        test_normal_histories, test_normal_embedding_models, test_normal_summaries = generate_embedding_by_round_new_samples(
            X_basic_embedding_model,
            X_original_normal_test,
            trees_model,
            5000000,
            embedding_models,
            n_classes,
            samples_embd_dim=samples_embd_dim,
            node_embd_dim=node_embd_dim
        )
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            test_normal_embedding_models[i].save(experiment_output / f"boosting_detector_test_normal_models_{i}.h5")
            # joblib.dump(test_normal_histories[i], experiment_output / 'boosting_detector_test_normal_histories_[i].jblib')
            # joblib.dump(test_normal_summaries[i], experiment_output / f'boosting_detector_test_normal_summaries_{i}.jblib', compress=1)

    test_normal_predictions = []
    for i in range(trees_model.n_estimators):
        pca_projector = round_classifiers[i][1]
        knn_classifier = round_classifiers[i][0]

        test_samples_embedding = pca_projector.transform(test_normal_embedding_models[i].layers[3].get_weights()[0])

        test_normal_predictions.append(knn_classifier.predict(test_samples_embedding.astype('float32')))

    if (experiment_output / 'boosting_detector_test_adv_models_0.h5').exists() and load_from_disk:
        test_adv_histories = []
        test_adv_embedding_models = []
        test_adv_summaries = []
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            # test_adv_histories.append(joblib.load(experiment_output / f'boosting_detector_test_adv_histories_{i}.jblib'))
            test_adv_embedding_models.append(
                keras.models.load_model(experiment_output / f'boosting_detector_test_adv_models_{i}.h5'))
            # test_adv_summaries.append(joblib.load(experiment_output / f'boosting_detector_test_adv_summaries_{i}.jblib'))
    else:
        test_adv_histories, test_adv_embedding_models, test_adv_summaries = generate_embedding_by_round_new_samples(
            X_basic_embedding_model,
            X_original_adv_test,
            trees_model,
            5000000,
            embedding_models,
            n_classes,
            samples_embd_dim=samples_embd_dim,
            node_embd_dim=node_embd_dim
        )
        for i in range(config['xgboost_model']['model_params']['n_estimators']):
            test_adv_embedding_models[i].save(experiment_output / f"boosting_detector_test_adv_models_{i}.h5")
            # joblib.dump(test_adv_histories[i], experiment_output / 'boosting_detector_test_adv_histories_[i].jblib')
            # joblib.dump(test_adv_summaries[i], experiment_output / f'boosting_detector_test_adv_summaries_{i}.jblib', compress=1)

    test_adv_predictions = []
    for i in range(trees_model.n_estimators):
        pca_projector = round_classifiers[i][1]
        knn_classifier = round_classifiers[i][0]

        test_samples_embedding = pca_projector.transform(test_adv_embedding_models[i].layers[3].get_weights()[0])

        test_adv_predictions.append(knn_classifier.predict(test_samples_embedding.astype('float32')))

    # log_likelihoods_normal = []
    # for i in range(np.array(train_predictions).shape[1]):
    #     log_likelihoods_normal.append(
    #         calculate_sample_switching_bayesian_log_likelihood(np.array(train_predictions)[:, i], train_probabilities))
    #
    # log_likelihoods_adv = []
    # for i in range(np.array(test_predictions).shape[1]):
    #     log_likelihoods_adv.append(
    #         calculate_sample_switching_bayesian_log_likelihood(np.array(test_predictions)[:, i], test_probabilities))
    #
    # train_true = np.zeros(len(log_likelihoods_train))
    # normal_pred = log_likelihoods_train
    # adv_true = np.ones(len(log_likelihoods_test))
    # adv_pred = log_likelihoods_test


def _check_simple_detector(X_original_normal_train, X_original_normal_test, X_original_adv_train, X_original_adv_test,
                           embedding_normal_train_model, embedding_normal_test_model, embedding_adv_train_model,
                           embedding_adv_test_model, experiment_output):
    """
    Train a simple adversarial samples detector for a decision trees ensembles

    :param X_original_normal_train:
        Dataset features for normal samples to train the detector
    :param X_original_normal_test:
        Dataset features for normal samples to test the detector
    :param X_original_adv_train:
        Dataset features for adversarial samples to train the detector
    :param X_original_adv_test:
        Dataset features for adversarial samples to test the detector
    :param embedding_normal_train_model:
    :param embedding_normal_test_model:
    :param embedding_adv_train_model:
    :param embedding_adv_test_model:
    :param experiment_output:
    :return:
    """
    detector_results = {}

    detector_results['sizes_normal_train'] = len(X_original_normal_train)
    detector_results['sizes_normal_test'] = len(X_original_normal_test)
    detector_results['sizes_adv_train'] = len(X_original_adv_train)
    detector_results['sizes_adv_test'] = len(X_original_adv_test)

    print(f"Number of normal samples train : {len(X_original_normal_train)}")
    print(f"Number of normal samples test : {len(X_original_normal_test)}")
    print(f"Number of adv samples train : {len(X_original_adv_train)}")
    print(f"Number of adv samples test : {len(X_original_adv_test)}")

    y_original_noraml_and_adv_train = np.hstack(
        [np.zeros(len(X_original_normal_train)), np.ones(len(X_original_adv_train))])
    X_original_normal_and_adv_train = np.vstack([X_original_normal_train, X_original_adv_train])

    y_original_noraml_and_adv_test = np.hstack(
        [np.zeros(len(X_original_normal_test)), np.ones(len(X_original_adv_test))])
    X_original_normal_and_adv_test = np.vstack([X_original_normal_test, X_original_adv_test])

    original_adv_detector = FaissKNNClassifier(n_neighbors=10)
    original_adv_detector.fit(X_original_normal_and_adv_train.astype('float32'),
                              y_original_noraml_and_adv_train.astype('float32'))

    original_train_predictions = original_adv_detector.predict(X_original_normal_and_adv_train.astype('float32'))
    original_train_probas = original_adv_detector.predict_proba(X_original_normal_and_adv_train.astype('float32'))[:, 1]
    original_test_predictions = original_adv_detector.predict(X_original_normal_and_adv_test.astype('float32'))
    original_test_proba = original_adv_detector.predict_proba(X_original_normal_and_adv_test.astype('float32'))[:, 1]

    detector_results['original_train_predicted_adv'] = np.count_nonzero(original_train_predictions)
    detector_results['original_train_recall'] = recall_score(y_original_noraml_and_adv_train,
                                                             original_train_predictions)
    detector_results['original_train_precision'] = precision_score(y_original_noraml_and_adv_train,
                                                                   original_train_predictions)
    detector_results['original_train_rocauc'] = roc_auc_score(y_original_noraml_and_adv_train, original_train_probas)

    detector_results['original_test_predicted_adv'] = np.count_nonzero(original_test_predictions)
    detector_results['original_test_recall'] = recall_score(y_original_noraml_and_adv_test, original_test_predictions)
    detector_results['original_test_precision'] = precision_score(y_original_noraml_and_adv_test,
                                                                  original_test_predictions)
    detector_results['original_test_rocauc'] = roc_auc_score(y_original_noraml_and_adv_test, original_test_proba)

    print('\n\n')
    print(f"Original Train Predicted Adv: {np.count_nonzero(original_train_predictions)}")
    print(f"Original Train Recall: {recall_score(y_original_noraml_and_adv_train, original_train_predictions)}")
    print(f"Original Train Precision: {precision_score(y_original_noraml_and_adv_train, original_train_predictions)}")
    print(f"Original Train AUC: {roc_auc_score(y_original_noraml_and_adv_train, original_train_probas)}")
    print('\n\n')
    print(f"Original Test Predicted Adv: {np.count_nonzero(original_test_predictions)}")
    print(f"Original Test Recall: {recall_score(y_original_noraml_and_adv_test, original_test_predictions)}")
    print(f"Original Test Precision: {precision_score(y_original_noraml_and_adv_test, original_test_predictions)}")
    print(f"Original Test AUC: {roc_auc_score(y_original_noraml_and_adv_test, original_test_proba)}")
    print('\n\n')

    X_embedded_normal_train = embedding_normal_train_model.layers[3].get_weights()[0]
    X_embedded_normal_test = embedding_normal_test_model.layers[3].get_weights()[0]
    X_embedded_adv_train = embedding_adv_train_model.layers[3].get_weights()[0]
    X_embedded_adv_test = embedding_adv_test_model.layers[3].get_weights()[0]

    y_embedded_noraml_and_adv_train = np.hstack(
        [np.zeros(len(X_embedded_normal_train)), np.ones(len(X_embedded_adv_train))])
    X_embedded_normal_and_adv_train = np.vstack([X_embedded_normal_train, X_embedded_adv_train])

    y_embedded_noraml_and_adv_test = np.hstack(
        [np.zeros(len(X_embedded_normal_test)), np.ones(len(X_embedded_adv_test))])
    X_embedded_normal_and_adv_test = np.vstack([X_embedded_normal_test, X_embedded_adv_test])

    embedded_adv_detector = FaissKNNClassifier(n_neighbors=10)
    embedded_adv_detector.fit(X_embedded_normal_and_adv_train.astype('float32'),
                              y_embedded_noraml_and_adv_train.astype('float32'))

    embedded_train_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_train.astype('float32'))
    embedded_train_probas = embedded_adv_detector.predict_proba(X_embedded_normal_and_adv_train.astype('float32'))[:, 1]
    embedded_test_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_test.astype('float32'))
    embedded_test_probas = embedded_adv_detector.predict_proba(X_embedded_normal_and_adv_test.astype('float32'))[:, 1]

    detector_results['embedded_train_predicted_adv'] = np.count_nonzero(embedded_train_predictions)
    detector_results['embedded_train_recall'] = recall_score(y_embedded_noraml_and_adv_train,
                                                             embedded_train_predictions)
    detector_results['embedded_train_precision'] = precision_score(y_embedded_noraml_and_adv_train,
                                                                   embedded_train_predictions)
    detector_results['embedded_train_rocauc'] = roc_auc_score(y_embedded_noraml_and_adv_train, embedded_train_probas)

    detector_results['embedded_test_predicted_adv'] = np.count_nonzero(embedded_test_predictions)
    detector_results['embedded_test_recall'] = recall_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)
    detector_results['embedded_test_precision'] = precision_score(y_embedded_noraml_and_adv_test,
                                                                  embedded_test_predictions)
    detector_results['embedded_test_rocauc'] = roc_auc_score(y_embedded_noraml_and_adv_test, embedded_test_probas)

    print('\n\n')
    print(f"Embedded Train Predicted Adv: {np.count_nonzero(embedded_train_predictions)}")
    print(f"Embedded Train Recall: {recall_score(y_embedded_noraml_and_adv_train, embedded_train_predictions)}")
    print(f"Embedded Train Precision: {precision_score(y_embedded_noraml_and_adv_train, embedded_train_predictions)}")
    print(f"Embedded Train AUC: {roc_auc_score(y_embedded_noraml_and_adv_train, embedded_train_probas)}")
    print('\n\n')
    print(f"Embedded Test Predicted Adv: {np.count_nonzero(embedded_test_predictions)}")
    print(f"Embedded Test Recall: {recall_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)}")
    print(f"Embedded Test Precision: {precision_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)}")
    print(f"Embedded Test AUC: {roc_auc_score(y_embedded_noraml_and_adv_test, embedded_test_probas)}")
    print('\n\n')

    joblib.dump(detector_results, experiment_output / 'detector_results.jblib', compress=1)


def dataset_spliter(X, y, need_test=True, X_test=None, y_test=None):
    """
    Given a dataset the function is splitting it to several different sets based on a constant ratios

    :param X: np.ndarray
        Dataset features
    :param y: np.ndarray
        Dataset labels
    :param need_test: bool
        If False a final test set is sampled from the data, if True the test set given in X_test and y_test is used.
    :param X_test:
        Optional parameter for a final test set features
    :param y_test:
        Optional parameter for a final test set labels
    :return:
    dict
        A dictionary with different sets based on the constants ratios
    """
    if need_test:
        dataset_split_portions = {
            'knn_org_perf': 0.01,
            'xgboost': 0.25,
            'embedding_model': 0.25,
            'detector_normal_train': 0.17,
            'detector_adv_train': 0.17,
            'detector_normal_test': 0.05,
            'detector_adv_test': 0.05,
            'final_test': 0.05
        }
    else:
        dataset_split_portions = {
            'knn_org_perf': 0.01,
            'xgboost': 0.30,
            'embedding_model': 0.35,
            'detector_normal_train': 0.095,
            'detector_adv_train': 0.095,
            'detector_normal_test': 0.075,
            'detector_adv_test': 0.075,
        }
        X_final_test = X_test
        y_final_test = y_test

    # KNN original performance dataset - 5%
    X_knn_org_perf, X_rest, y_knn_org_perf, y_rest = train_test_split(X, y, shuffle=True,
                                                                      train_size=dataset_split_portions[
                                                                          'knn_org_perf'])

    # XGBoost classifier training dataset - 35%
    xgboost_previous = dataset_split_portions['knn_org_perf']
    X_xgboost, X_rest, y_xgboost, y_rest = train_test_split(X_rest, y_rest, shuffle=True, train_size=(
            dataset_split_portions['xgboost'] / (1 - xgboost_previous)))

    # Train embedding model dataset - 30%
    embedding_previous = xgboost_previous + dataset_split_portions['xgboost']
    X_embedding, X_rest, y_embedding, y_rest = train_test_split(X_rest, y_rest, shuffle=True, train_size=(
            dataset_split_portions['embedding_model'] / (1 - embedding_previous)))

    # Normal dataset to train the detector - 9%
    detector_normal_train_previous = embedding_previous + dataset_split_portions['embedding_model']
    X_detector_train_normal, X_rest, y_detector_train_normal, y_rest = train_test_split(X_rest, y_rest,
                                                                                        shuffle=True,
                                                                                        train_size=
                                                                                        dataset_split_portions[
                                                                                            'detector_normal_train'] / (
                                                                                                1 - detector_normal_train_previous))

    # Normal dataset to test the detector - 1%
    detector_normal_test_previous = detector_normal_train_previous + dataset_split_portions['detector_normal_train']
    X_detector_test_normal, X_rest, y_detector_test_normal, y_rest = train_test_split(X_rest, y_rest, shuffle=True,
                                                                                      train_size=
                                                                                      dataset_split_portions[
                                                                                          'detector_normal_test'] / (
                                                                                              1 - detector_normal_test_previous))

    # Adversarial dataset to train the detector - 9%
    detector_adv_train_previous = detector_normal_test_previous + dataset_split_portions['detector_normal_test']
    X_detector_train_adv, X_rest, y_detector_train_adv, y_rest = train_test_split(X_rest, y_rest, shuffle=True,
                                                                                  train_size=dataset_split_portions[
                                                                                                 'detector_adv_train'] / (
                                                                                                     1 - detector_adv_train_previous))

    # Adversarial dataset to test the detector - 1%
    if need_test:
        detector_adv_test_previous = detector_adv_train_previous + dataset_split_portions['detector_adv_train']
        X_detector_test_adv, X_final_test, y_detector_test_adv, y_final_test = train_test_split(X_rest, y_rest,
                                                                                                shuffle=True,
                                                                                                train_size=
                                                                                                dataset_split_portions[
                                                                                                    'detector_adv_test'] / (
                                                                                                        1 - detector_adv_test_previous))
    else:
        X_detector_test_adv = X_rest
        y_detector_test_adv = y_rest

    return {
        'knn_org_perf': (X_knn_org_perf, y_knn_org_perf),
        'xgboost': (X_xgboost, y_xgboost),
        'embedding_model': (X_embedding, y_embedding),
        'detector_normal_train': (X_detector_train_normal, y_detector_train_normal),
        'detector_normal_test': (X_detector_test_normal, y_detector_test_normal),
        'detector_adv_train': (X_detector_train_adv, y_detector_train_adv),
        'detector_adv_test': (X_detector_test_adv, y_detector_test_adv),
        'final_test': (X_final_test, y_final_test)
    }


if __name__ == '__main__':
    import numpy as np

    parser = argparse.ArgumentParser(description='Adversarial Attack')
    parser.add_argument('dataset', type=str, help='The dataset to load')
    parser.add_argument('attack_type', type=str, help='Attack type')
    parser.add_argument('norm', type=str, help='Attack norm')
    parser.add_argument('detector_type', type=str, help="Detector type")
    parser.add_argument('sampled_embed_dim', type=int, help="Samples embedding size", default=100)
    parser.add_argument('nodes_embed_dim', type=int, help="Nodes embedding size", default=50)
    parser.add_argument('modeltype', type=str, help='Model type (xgboost|rf)')
    parser.add_argument('--purpose', type=str, help="Execution purpose (adv|emb)", default="adv")

    args = parser.parse_args()

    dataset = args.dataset
    norm = np.inf if args.norm == 'inf' else int(args.norm)
    attack_type = args.attack_type
    detector_type = args.detector_type
    samples_embed_dim = args.sampled_embed_dim
    nodes_embed_dim = args.nodes_embed_dim
    model_type = args.modeltype
    exec_purpose = args.purpose

    config = template_config
    config['dataset']['dataset_name'] = dataset
    config['adv_samples']['norm_type'] = norm
    config['adv_samples']['search_mode'] = attack_type
    config['detector_adv_samples']['norm_type'] = norm
    config['detector_adv_samples']['search_mode'] = attack_type
    config['detector_config']['type'] = detector_type
    config['embed_model']['samples_embd_dim'] = samples_embed_dim
    config['embed_model']['nodes_embd_dim'] = nodes_embed_dim
    config['xgboost_model']['model_type'] = 'XGBoost' if model_type == 'xgboost' else 'RandomForest'
    config['exec_purpose'] = exec_purpose

    if model_type == "rf":
        config['output']['output_path_parent'] = 'thesis_results_rf'

    trees_model_v1 = full_experiment_cycle(config, load_from_disk=True)


#
#
# # Latex print
# def metric_chunk_for_line(embedded_val, original_val):
#     if embedded_val > original_val:
#         return '\t& \t\\underline{\\textbf{' + f'{round(embedded_val, 3)}' + '}}\t &\t' + f'{round(original_val, 3)}'
#     elif original_val > embedded_val:
#         return '\t &\t ' + f'{round(embedded_val, 3)}' + '\t& \t\\underline{\\textbf{' + f'{round(original_val, 3)}' + '}}'
#     else:
#         return '\t &\t ' + f'{round(embedded_val, 3)}' + '\t& \t' + f'{round(original_val, 3)}' + ''
#
#
# def extract_perf_dict(res_root, dataset, attack, norm, k=10):
#     res_path = res_root / f'{dataset}_{attack}_{norm}'
#
#     dataset_splitted = joblib.load(res_path / 'dataset_splitted.jblib')
#
#     X_original_normal_train = dataset_splitted['detector_normal_train'][0]
#     X_original_normal_test = dataset_splitted['detector_normal_test'][0]
#
#     train_adv_samples = joblib.load(res_path / f'train_detector_adv_results.jblib')
#     test_adv_samples = joblib.load(res_path / f'test_detector_adv_results.jblib')
#
#     X_adv_train = []
#     for sample in train_adv_samples:
#         if train_adv_samples[sample]['model_classify_correct'] and train_adv_samples[sample].get('adv_succ', False):
#             X_adv_train.append(train_adv_samples[sample]['adv_vector'])
#     X_original_adv_train = np.vstack(X_adv_train)
#
#     X_adv_test = []
#     for sample in test_adv_samples:
#         if test_adv_samples[sample]['model_classify_correct'] and test_adv_samples[sample].get('adv_succ', False):
#             X_adv_test.append(test_adv_samples[sample]['adv_vector'])
#     X_original_adv_test = np.vstack(X_adv_test)
#
#     embedding_normal_train_model = keras.models.load_model(res_path / f"embedding_detector_normal_train_model.h5")
#     embedding_normal_test_model = keras.models.load_model(res_path / f"embedding_detector_normal_test_model.h5")
#     embedding_adv_train_model = keras.models.load_model(res_path / f"embedding_detector_adv_train_model.h5")
#     embedding_adv_test_model = keras.models.load_model(res_path / f"embedding_detector_adv_test_model.h5")
#
#     detector_results = {}
#
#     detector_results['sizes_normal_train'] = len(X_original_normal_train)
#     detector_results['sizes_normal_test'] = len(X_original_normal_test)
#     detector_results['sizes_adv_train'] = len(X_original_adv_train)
#     detector_results['sizes_adv_test'] = len(X_original_adv_test)
#
#     y_original_noraml_and_adv_train = np.hstack(
#         [np.zeros(len(X_original_normal_train)), np.ones(len(X_original_adv_train))])
#     X_original_normal_and_adv_train = np.vstack([X_original_normal_train, X_original_adv_train])
#
#     y_original_noraml_and_adv_test = np.hstack(
#         [np.zeros(len(X_original_normal_test)), np.ones(len(X_original_adv_test))])
#     X_original_normal_and_adv_test = np.vstack([X_original_normal_test, X_original_adv_test])
#
#     original_adv_detector = FaissKNNClassifier(n_neighbors=k)
#     original_adv_detector.fit(X_original_normal_and_adv_train.astype('float32'),
#                               y_original_noraml_and_adv_train.astype('float32'))
#
#     original_train_predictions = original_adv_detector.predict(X_original_normal_and_adv_train.astype('float32'))
#     original_train_probas = original_adv_detector.predict_proba(X_original_normal_and_adv_train.astype('float32'))[:, 1]
#     original_test_predictions = original_adv_detector.predict(X_original_normal_and_adv_test.astype('float32'))
#     original_test_proba = original_adv_detector.predict_proba(X_original_normal_and_adv_test.astype('float32'))[:, 1]
#
#     detector_results['original_train_predicted_adv'] = np.count_nonzero(original_train_predictions)
#     detector_results['original_train_recall'] = recall_score(y_original_noraml_and_adv_train,
#                                                              original_train_predictions)
#     detector_results['original_train_precision'] = precision_score(y_original_noraml_and_adv_train,
#                                                                    original_train_predictions)
#     detector_results['original_train_rocauc'] = roc_auc_score(y_original_noraml_and_adv_train, original_train_probas)
#
#     detector_results['original_test_predicted_adv'] = np.count_nonzero(original_test_predictions)
#     detector_results['original_test_recall'] = recall_score(y_original_noraml_and_adv_test, original_test_predictions)
#     detector_results['original_test_precision'] = precision_score(y_original_noraml_and_adv_test,
#                                                                   original_test_predictions)
#     detector_results['original_test_rocauc'] = roc_auc_score(y_original_noraml_and_adv_test, original_test_proba)
#     detector_results['original_test_ap'] = average_precision_score(y_original_noraml_and_adv_test, original_test_proba)
#     detector_results['original_test_label'] = y_original_noraml_and_adv_test
#     detector_results['original_test_proba'] = original_test_proba
#
#     X_embedded_normal_train = embedding_normal_train_model.layers[3].get_weights()[0]
#     X_embedded_normal_test = embedding_normal_test_model.layers[3].get_weights()[0]
#     X_embedded_adv_train = embedding_adv_train_model.layers[3].get_weights()[0]
#     X_embedded_adv_test = embedding_adv_test_model.layers[3].get_weights()[0]
#
#     y_embedded_noraml_and_adv_train = np.hstack(
#         [np.zeros(len(X_embedded_normal_train)), np.ones(len(X_embedded_adv_train))])
#     X_embedded_normal_and_adv_train = np.vstack([X_embedded_normal_train, X_embedded_adv_train])
#
#     y_embedded_noraml_and_adv_test = np.hstack(
#         [np.zeros(len(X_embedded_normal_test)), np.ones(len(X_embedded_adv_test))])
#     X_embedded_normal_and_adv_test = np.vstack([X_embedded_normal_test, X_embedded_adv_test])
#
#     embedded_adv_detector = FaissKNNClassifier(n_neighbors=k)
#     embedded_adv_detector.fit(X_embedded_normal_and_adv_train.astype('float32'),
#                               y_embedded_noraml_and_adv_train.astype('float32'))
#
#     embedded_train_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_train.astype('float32'))
#     embedded_train_probas = embedded_adv_detector.predict_proba(X_embedded_normal_and_adv_train.astype('float32'))[:, 1]
#     embedded_test_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_test.astype('float32'))
#     embedded_test_probas = embedded_adv_detector.predict_proba(X_embedded_normal_and_adv_test.astype('float32'))[:, 1]
#
#     detector_results['embedded_train_predicted_adv'] = np.count_nonzero(embedded_train_predictions)
#     detector_results['embedded_train_recall'] = recall_score(y_embedded_noraml_and_adv_train,
#                                                              embedded_train_predictions)
#     detector_results['embedded_train_precision'] = precision_score(y_embedded_noraml_and_adv_train,
#                                                                    embedded_train_predictions)
#     detector_results['embedded_train_rocauc'] = roc_auc_score(y_embedded_noraml_and_adv_train, embedded_train_probas)
#
#     detector_results['embedded_test_predicted_adv'] = np.count_nonzero(embedded_test_predictions)
#     detector_results['embedded_test_recall'] = recall_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)
#     detector_results['embedded_test_precision'] = precision_score(y_embedded_noraml_and_adv_test,
#                                                                   embedded_test_predictions)
#     detector_results['embedded_test_rocauc'] = roc_auc_score(y_embedded_noraml_and_adv_test, embedded_test_probas)
#     detector_results['embedded_test_ap'] = average_precision_score(y_embedded_noraml_and_adv_test, embedded_test_probas)
#     detector_results['embedded_test_label'] = y_embedded_noraml_and_adv_test
#     detector_results['embedded_test_proba'] = embedded_test_probas
#
#     attacks_mapping = {'leaftuple': 'Leaf-Tuple', 'cube': 'Cube', 'hsja': 'HJSA', 'opt': 'OPT', 'signopt': 'Sign-OPT'}
#     detector_results['attack_name'] = attacks_mapping[attack]
#
#     datasets_mapping = {'sensorless': 'Sensorless', 'codrna': 'cod-rna', 'wind': 'wind', 'speech': 'speech',
#                         'spambase': 'spambase', 'waveform': 'waveform', 'kc1': 'kc1', 'dry_bean': 'dry_bean',
#                         'banknote': 'banknote', 'shuttle': 'shuttle', 'electricity': 'electricity', 'adult': 'adult',
#                         'voice': 'gender-by-voice', 'breast_cancer': 'breast-cancer', 'covtype': 'covtype',
#                         'diabetes': 'diabetes', 'mnist26': 'MNIST2-6', 'ijcnn1': 'ijcnn', 'mnist': 'MNIST',
#                         'fashion': 'Fashion-MNIST', 'webspam': 'webspam', 'covtype': 'covtype'}
#     detector_results['dataset_name'] = datasets_mapping[dataset]
#
#     detector_results['norm_name'] = '2' if norm == '2' else '$\infty$'
#
#     return detector_results
#
#
# def print_roc_curve(res_info, save_images=False, image_path=None):
#     display = RocCurveDisplay.from_predictions(res_info['embedded_test_label'], res_info['embedded_test_proba'],
#                                                name="Embedded")
#     RocCurveDisplay.from_predictions(res_info['original_test_label'], res_info['original_test_proba'], name="Original",
#                                      ax=display.ax_)
#     _ = display.ax_.set_title(
#         f"ROC curve | Attack: {res_info['attack_name']} | Dataset: {res_info['dataset_name']} | Norm: {res_info['norm_name']}")
#     if save_images:
#         plt.savefig(image_path, dpi=300)
#
#
# def print_pr_curve(res_info, save_images=False, image_path=None):
#     display = PrecisionRecallDisplay.from_predictions(res_info['embedded_test_label'], res_info['embedded_test_proba'],
#                                                       name="Embedded")
#     PrecisionRecallDisplay.from_predictions(res_info['original_test_label'], res_info['original_test_proba'],
#                                             name="Original", ax=display.ax_)
#     _ = display.ax_.set_title(
#         f"Precision-Recall curve | Attack: {res_info['attack_name']} | Dataset: {res_info['dataset_name']} | Norm: {res_info['norm_name']}")
#     if save_images:
#         plt.savefig(image_path, dpi=300)
#
#
# def print_results_line(res_root, dataset, attack, norms=('2', 'inf'), k=10):
#     latex_line = ""
#
#     for norm in norms:
#         detector_results = extract_perf_dict(res_root, dataset, attack, norm, k)
#
#         latex_line += metric_chunk_for_line(detector_results['embedded_test_ap'], detector_results['original_test_ap'])
#         latex_line += metric_chunk_for_line(detector_results['embedded_test_rocauc'],
#                                             detector_results['original_test_rocauc'])
#
#     print(latex_line)
#
#
# def print_total_results(res_root, dataset, attack, norms=('2', 'inf'), k=10):
#     print_results_line(res_root, dataset, attack, norms, k)
#
#     for norm in norms:
#         res_info = extract_perf_dict(res_root, dataset, attack, norm, k)
#
#         print_roc_curve(res_info, True, res_root / f'{dataset}_{attack}_{norm}' / f'roc_{dataset}_{attack}_{norm}.png')
#         print_pr_curve(res_info, True, res_root / f'{dataset}_{attack}_{norm}' / f'pr_{dataset}_{attack}_{norm}.png')

import numpy as np
import xgboost
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
def check_oc_score(max_from_normal=100):
    xgb = xgboost.XGBClassifier()
    xgb.load_model('trees_model_sklearn_version.json')
    dataset_splitted = joblib.load('dataset_splitted.jblib')
    x_train, y_train = dataset_splitted['xgboost']
    train_leafs = xgb.apply(x_train)

    test_det_res = joblib.load('test_detector_adv_results.jblib')
    X_adv_train = []
    for sample in test_det_res:
        if test_det_res[sample]['model_classify_correct'] and test_det_res[sample].get('adv_succ', False):
            X_adv_train.append(test_det_res[sample]['adv_vector'])
    X_adv_stacked_train = np.vstack(X_adv_train)

    normal_minimals = []
    for x in tqdm(dataset_splitted['final_test'][0][:max_from_normal, :], total=max_from_normal):
        normal_minimals.append(min((xgb.apply(x.reshape(1, -1)) != train_leafs).sum(axis=1)))
    adv_minimals = []
    for x in tqdm(X_adv_stacked_train, total=len(X_adv_stacked_train)):
        adv_minimals.append(min((xgb.apply(x.reshape(1, -1)) != train_leafs).sum(axis=1)))

    return normal_minimals, adv_minimals

# normal_minimals, adv_minimals = check_oc_score()
# labels = np.hstack([np.zeros(len(normal_minimals)) , np.ones(len(adv_minimals))])
# preds= np.array([0 if nm <= 14 else 1 for nm in normal_minimals] + [0 if am <= 14 else 1 for am in adv_minimals])
# from sklearn.metrics import roc_auc_score
# roc_auc_score(labels, preds)
# preds= np.array([0 if nm <= 14 else 1 for nm in normal_minimals] + [0 if am <= 10 else 1 for am in adv_minimals])
# roc_auc_score(labels, preds)