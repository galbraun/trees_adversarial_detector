import json
import pathlib

import faiss
import holoviews as hv
import joblib
import numpy as np
import umap
import xgboost
from bokeh.io import save as bokeh_save
from bokeh.resources import INLINE as bokeh_resources_inline
from holoviews import opts
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
import pandas as pd
import argparse

from trees_adversarial_detector.attack import generate_adv_samples
from trees_adversarial_detector.datasets import load_dataset
from trees_adversarial_detector.embedding_proc import extract_embedding_dataset, train_embedding, \
    extract_new_samples_embedding_dataset, train_embedding_new_set
from trees_adversarial_detector.evaluation import FaissKNeighbors
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
            "num_attack_per_point": 10,
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
            "num_attack_per_point": 10,
            "norm_type": 2,
            "search_mode": "signopt"
        },
    'embed_model':
        {
            'main_model_dataset_size': 250000000,
            'adv_repr_dataset_size': 10000000,
            'test_repr_dataset_size': 10000000,
            'detector_normal_train_repr_dataset_size': 10000000,
            'detector_normal_test_repr_dataset_size': 10000000,
            'detector_adv_train_repr_dataset_size': 10000000,
            'detector_adv_test_repr_dataset_size': 10000000,
            'epochs_detector_normal_train_model': 2,
            'epochs_detector_normal_test_model': 2,
            'epochs_detector_adv_train_model': 2,
            'epochs_detector_adv_test_model': 2,
            'epochs_main_model': 1,
            'epochs_adv_model': 2,
            'epochs_test_model': 2,
            'samples_embd_dim': 20,
            'nodes_embd_dim': 10,
        }

}


def _load_dataset(config, experiment_output, load_from_disk):
    # Load dataset
    X, y = load_dataset(config['dataset']['dataset_name'])

    if type(X) == pd.DataFrame:
        X = X.values

    # Split dataset
    if (experiment_output / 'dataset_splitted.jblib').exists() and load_from_disk:
        # X_train = joblib.load(experiment_output / 'X_train.jblib')
        # X_test = joblib.load(experiment_output / 'X_test.jblib')
        #
        # y_train = joblib.load(experiment_output / 'y_train.jblib')
        # y_test = joblib.load(experiment_output / 'y_test.jblib')
        dataset_splitted = joblib.load(experiment_output / 'dataset_splitted.jblib')
    else:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['dataset_split']['test_size'])
        # joblib.dump(X_train, experiment_output / 'X_train.jblib')
        # joblib.dump(X_test, experiment_output / 'X_test.jblib')
        # joblib.dump(y_train, experiment_output / 'y_train.jblib')
        # joblib.dump(y_test, experiment_output / 'y_test.jblib')
        # dump_svmlight_file(X_train, y_train, experiment_output / 'train_svm_data.svmlight',
        #                    comment=f"{config['dataset']['dataset_name']} train original data")
        # dump_svmlight_file(X_test, y_test, experiment_output / 'test_svm_data.svmlight',
        #                    comment=f"{config['dataset']['dataset_name']} test original data")
        dataset_splitted = dataset_spliter(X, y)
        joblib.dump(dataset_splitted, experiment_output / 'dataset_splitted.jblib')
        for dataset_name in dataset_splitted.keys():
            dump_svmlight_file(dataset_splitted[dataset_name][0], dataset_splitted[dataset_name][1],
                               experiment_output / f'{dataset_name}_data.svmlight',
                               comment=f"{dataset_name} data")

    return dataset_splitted


def _original_repr_knn_performance(X_train, X_test, y_train, y_test, config, experiment_output):
    orignal_knn_k_low = config['original_features_knn']['k_low']
    orignal_knn_k_high = config['original_features_knn']['k_high']
    original_knn_k_total = orignal_knn_k_high - orignal_knn_k_low
    original_acc = []
    original_recall = []
    original_precision = []

    for i in tqdm(range(orignal_knn_k_low, orignal_knn_k_high), total=original_knn_k_total,
                  desc="Checking KNN for original feature space for different Ks"):
        neigh = FaissKNeighbors(k=i, faiss=faiss)
        neigh.fit(X_train.astype('float32'), y_train.astype('float32'))
        test_preds = neigh.predict(X_test.astype('float32'))
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

    train_proba = trees_model.predict_proba(X_train)[:, 1]
    test_proba = trees_model.predict_proba(X_test)[:, 1]

    fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_proba)
    fpr_test, tpr_test, thresh_test = roc_curve(y_test, test_proba)

    train_roc = hv.Curve(zip(fpr_train, tpr_train), label=f'train_roc (AUC = {roc_auc_score(y_train, train_proba)})')
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


def _generate_adv_samples(config, experiment_output, num_features, load_from_disk, train_data_filename,
                          test_data_filename, set_name='adv'):
    if (experiment_output / f'train_{set_name}_results.jblib').exists() and load_from_disk:
        train_adv_samples = joblib.load(experiment_output / f'train_{set_name}_results.jblib')
        test_adv_samples = joblib.load(experiment_output / f'test_{set_name}_results.jblib')
    else:
        adv_config = config[f'{set_name}_samples']
        adv_config['num_features'] = num_features
        adv_config['model'] = str(experiment_output / 'trees_model.json')

        adv_config['inputs'] = str(experiment_output / train_data_filename)
        with open(experiment_output / 'train_attack_config.json', 'w') as json_file:
            json.dump(adv_config, json_file)
        train_adv_samples, train_final_statistics = generate_adv_samples(experiment_output / 'train_attack_config.json')
        joblib.dump(train_adv_samples, experiment_output / f'train_{set_name}_results.jblib')
        joblib.dump(train_final_statistics, experiment_output / f'train_{set_name}_statistics.jblib')

        adv_config['inputs'] = str(experiment_output / test_data_filename)
        with open(experiment_output / 'test_attack_config.json', 'w') as json_file:
            json.dump(adv_config, json_file)
        test_adv_samples, test_final_statistics = generate_adv_samples(experiment_output / 'test_attack_config.json')
        joblib.dump(test_adv_samples, experiment_output / f'test_{set_name}_results.jblib')
        joblib.dump(test_final_statistics, experiment_output / f'test_{set_name}_statistics.jblib')

    X_adv_train = []
    for sample in train_adv_samples:
        if train_adv_samples[sample]['model_classify_correct'] and train_adv_samples[sample]['adv_succ']:
            X_adv_train.append(train_adv_samples[sample]['adv_vector'])
    X_adv_stacked_train = np.vstack(X_adv_train)

    X_adv_test = []
    for sample in test_adv_samples:
        if test_adv_samples[sample]['model_classify_correct'] and test_adv_samples[sample]['adv_succ']:
            X_adv_test.append(test_adv_samples[sample]['adv_vector'])
    X_adv_stacked_test = np.vstack(X_adv_test)

    return X_adv_stacked_train, X_adv_stacked_test


def _extract_main_embedding_dataset(config, experiment_output, X, trees_model, load_from_disk):
    if (experiment_output / 'embedding_dataset_X.jblib').exists() and load_from_disk:
        embedding_X = joblib.load(experiment_output / 'embedding_dataset_X.jblib')
        embedding_y = joblib.load(experiment_output / 'embedding_dataset_y.jblib')
        num_nodes = joblib.load(experiment_output / 'main_num_nodes.jblib')
    else:
        embedding_X, embedding_y, num_nodes = extract_embedding_dataset(X, trees_model,
                                                                        config['embed_model'][
                                                                            'main_model_dataset_size'])
        joblib.dump(embedding_X, experiment_output / 'embedding_dataset_X.jblib')
        joblib.dump(embedding_y, experiment_output / 'embedding_dataset_y.jblib')
        joblib.dump(num_nodes, experiment_output / 'main_num_nodes.jblib')

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
        joblib.dump(summary, experiment_output / 'embedding_model_summary.jblib')

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
        joblib.dump(embedding_X_adv, experiment_output / f'embedding_{dataset_name}_dataset_X.jblib')
        joblib.dump(embedding_y_adv, experiment_output / f'embedding_{dataset_name}_dataset_y.jblib')
        joblib.dump(num_nodes_adv, experiment_output / f'{dataset_name}_num_nodes.jblib')

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
        joblib.dump(summary_adv, experiment_output / f'embedding_{dataset_name}_model_summary.jblib')

    return embedding_model_adv, summary_adv


def _compare_representations(X_train, X_train_embed_model, X_test, X_test_embed_model, y_train, y_test, config,
                             experiment_output):
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
        neigh_embed = FaissKNeighbors(k=i, faiss=faiss)
        neigh_embed.fit(X_train_embed.astype('float32'), y_train.astype('float32'))
        embed_preds = neigh_embed.predict(X_test_embed.astype('float32'))

        neigh_orig = FaissKNeighbors(k=i, faiss=faiss)
        neigh_orig.fit(X_train.astype('float32'), y_train.astype('float32'))
        orig_preds = neigh_orig.predict(X_test.astype('float32'))

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
                                  y_train, y_test):
    train_samples_embedding = embedding_model.layers[3].get_weights()[0]
    adv_samples_weights = embedding_model_adv.layers[3].get_weights()[0]
    test_samples_weights = embedding_model_test.layers[3].get_weights()[0]

    all_data = np.vstack([train_samples_embedding, test_samples_weights, adv_samples_weights])
    indexes = np.hstack([y_train, y_test, 2 * np.ones(adv_samples_weights.shape[0])])

    all_data_in_2d = _reduce_data_to_2d(all_data)

    normal_negative_scatter = hv.Scatter(all_data_in_2d[(indexes == 0).astype(bool), :],
                                         label='Normal negative samples').opts(size=5)
    normal_positive_scatter = hv.Scatter(all_data_in_2d[(indexes == 1).astype(bool), :],
                                         label='Normal positive samples').opts(size=5)
    adv_samples = hv.Scatter(all_data_in_2d[(indexes == 2).astype(bool), :], label='Adv samples').opts(
        size=10, color='green')

    all_scatters = (normal_negative_scatter * normal_positive_scatter * adv_samples).opts(
        opts.Scatter(tools=['hover'], width=1000, height=500))
    all_scatters = all_scatters.opts(legend_position='right',
                                     title='Normal and Adv sample 2d representation based on embedding')

    bokeh_plot = renderer.get_plot(all_scatters).state
    bokeh_save(bokeh_plot, experiment_output / 'embedding_2d_representations.html',
               resources=bokeh_resources_inline)


def full_experiment_cycle(config, load_from_disk=True):
    output_parent = pathlib.Path(config['output']['output_path_parent'])
    experiment_output = output_parent / f"{config['dataset']['dataset_name']}_{config['adv_samples']['search_mode']}_{config['adv_samples']['norm_type']}"

    if not experiment_output.exists():
        experiment_output.mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset")
    dataset_splitted = _load_dataset(config, experiment_output, load_from_disk)

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
                                                    load_from_disk, 'xgboost_data.svmlight', 'final_test_data.svmlight')

    # Extract adversarial dataset representation using original representation
    print("Extracting 2D representation with adversarial samples")
    _with_adv_dataset_2d_rep(experiment_output, dataset_splitted['xgboost'][0], dataset_splitted['final_test'][0],
                             X_adv_train, X_adv_test)

    # Extract embedding dataset
    print("Extracting dataset for embedding calculation")
    embedding_X, embedding_y, num_nodes = _extract_main_embedding_dataset(config, experiment_output,
                                                                          dataset_splitted['embedding_model'][0],
                                                                          trees_model, load_from_disk)

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
                                  dataset_splitted['embedding_model'][1], dataset_splitted['final_test'][1])

    # Extract representation comparison
    _compare_representations(dataset_splitted['embedding_model'][0], embedding_model, dataset_splitted['final_test'][0],
                             embedding_model_test, dataset_splitted['embedding_model'][1],
                             dataset_splitted['final_test'][1], config, experiment_output)

    # Prepare data for Adv detector
    print("Preparing adversarial samples")
    X_detector_adv_train, X_detector_adv_test = _generate_adv_samples(config, experiment_output,
                                                                      dataset_splitted['detector_adv_train'][0].shape[
                                                                          1],
                                                                      load_from_disk,
                                                                      'detector_adv_train_data.svmlight',
                                                                      'detector_adv_test_data.svmlight', 'detector_adv')

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
    embedding_model_detector_normal_test, summary_detector_normal_test = _extract_extra_samples_representations(config,
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
                                                                                                            len(X_detector_adv_train),
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
                                                                                                          len(X_detector_adv_test),
                                                                                                          dataset_splitted[
                                                                                                              'detector_adv_test'][
                                                                                                              0].shape[
                                                                                                              1],
                                                                                                          load_from_disk,
                                                                                                          'detector_adv_test')

    _check_simple_detector(dataset_splitted['detector_normal_train'][0], dataset_splitted['detector_normal_test'][0],
                           X_detector_adv_train, X_detector_adv_test,
                           embedding_model_detector_normal_train, embedding_model_detector_normal_test,
                           embedding_model_detector_adv_train,
                           embedding_model_detector_adv_test, experiment_output)

    # Evaluate Adv detector
    # check_simple_detector(config['dataset']['dataset_name'], experiment_output / config['dataset']['dataset_name'],
    #                       X.shape[1])

    # Boosting Detector
    pass


def _check_simple_detector(X_original_normal_train, X_original_normal_test, X_original_adv_train, X_original_adv_test,
                           embedding_normal_train_model, embedding_normal_test_model, embedding_adv_train_model,
                           embedding_adv_test_model, experiment_output):
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

    original_adv_detector = FaissKNeighbors(k=10, faiss=faiss)
    original_adv_detector.fit(X_original_normal_and_adv_train.astype('float32'),
                              y_original_noraml_and_adv_train.astype('float32'))

    original_train_predictions = original_adv_detector.predict(X_original_normal_and_adv_train.astype('float32'))
    original_test_predictions = original_adv_detector.predict(X_original_normal_and_adv_test.astype('float32'))

    detector_results['original_train_predicted_adv'] = np.count_nonzero(original_train_predictions)
    detector_results['original_train_recall'] = recall_score(y_original_noraml_and_adv_train,
                                                             original_train_predictions)
    detector_results['original_train_precision'] = precision_score(y_original_noraml_and_adv_train,
                                                                   original_train_predictions)
    detector_results['original_test_predicted_adv'] = np.count_nonzero(original_test_predictions)
    detector_results['original_test_recall'] = recall_score(y_original_noraml_and_adv_test, original_test_predictions)
    detector_results['original_test_precision'] = precision_score(y_original_noraml_and_adv_test,
                                                                  original_test_predictions)

    print('\n\n')
    print(f"Original Train Predicted Adv: {np.count_nonzero(original_train_predictions)}")
    print(f"Original Train Recall: {recall_score(y_original_noraml_and_adv_train, original_train_predictions)}")
    print(f"Original Train Precision: {precision_score(y_original_noraml_and_adv_train, original_train_predictions)}")
    print('\n\n')
    print(f"Original Test Predicted Adv: {np.count_nonzero(original_test_predictions)}")
    print(f"Original Test Recall: {recall_score(y_original_noraml_and_adv_test, original_test_predictions)}")
    print(f"Original Test Precision: {precision_score(y_original_noraml_and_adv_test, original_test_predictions)}")
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

    embedded_adv_detector = FaissKNeighbors(k=10, faiss=faiss)
    embedded_adv_detector.fit(X_embedded_normal_and_adv_train.astype('float32'),
                              y_embedded_noraml_and_adv_train.astype('float32'))

    embedded_train_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_train.astype('float32'))
    embedded_test_predictions = embedded_adv_detector.predict(X_embedded_normal_and_adv_test.astype('float32'))

    detector_results['embedded_train_predicted_adv'] = np.count_nonzero(embedded_train_predictions)
    detector_results['embedded_train_recall'] = recall_score(y_embedded_noraml_and_adv_train, embedded_train_predictions)
    detector_results['embedded_train_precision'] = precision_score(y_embedded_noraml_and_adv_train,
                                                                   embedded_train_predictions)
    detector_results['embedded_test_predicted_adv'] = np.count_nonzero(embedded_test_predictions)
    detector_results['embedded_test_recall'] = recall_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)
    detector_results['embedded_test_precision'] = precision_score(y_embedded_noraml_and_adv_test,
                                                                  embedded_test_predictions)

    print('\n\n')
    print(f"Embedded Train Predicted Adv: {np.count_nonzero(embedded_train_predictions)}")
    print(f"Embedded Train Recall: {recall_score(y_embedded_noraml_and_adv_train, embedded_train_predictions)}")
    print(f"Embedded Train Precision: {precision_score(y_embedded_noraml_and_adv_train, embedded_train_predictions)}")
    print('\n\n')
    print(f"Embedded Test Predicted Adv: {np.count_nonzero(embedded_test_predictions)}")
    print(f"Embedded Test Recall: {recall_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)}")
    print(f"Embedded Test Precision: {precision_score(y_embedded_noraml_and_adv_test, embedded_test_predictions)}")
    print('\n\n')

    joblib.dump(detector_results, experiment_output / 'detector_results.jblib')


def dataset_spliter(X, y, need_test=True, X_test=None, y_test=None):
    if need_test:
        dataset_split_portions = {
            'knn_org_perf': 0.05,
            'xgboost': 0.25,
            'embedding_model': 0.25,
            'detector_normal_train': 0.15,
            'detector_adv_train': 0.15,
            'detector_normal_test': 0.05,
            'detector_adv_test': 0.05,
            'final_test': 0.05
        }
    else:
        dataset_split_portions = {
            'knn_org_perf': 0.05,
            'xgboost': 0.40,
            'embedding_model': 0.35,
            'detector_normal_train': 0.05,
            'detector_adv_train': 0.05,
            'detector_normal_test': 0.05,
            'detector_adv_test': 0.05,
        }

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
    detector_adv_test_previous = detector_adv_train_previous + dataset_split_portions['detector_adv_train']
    X_detector_test_adv, X_final_test, y_detector_test_adv, y_final_test = train_test_split(X_rest, y_rest,
                                                                                            shuffle=True,
                                                                                            train_size=
                                                                                            dataset_split_portions[
                                                                                                'detector_adv_test'] / (
                                                                                                    1 - detector_adv_test_previous))

    # Final test set - 10%
    if not need_test:
        X_embedding = np.vstack([X_embedding, X_final_test])
        y_embedding = np.hstack([y_embedding, y_final_test])
        X_final_test = X_test
        y_final_test = y_test

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

    args = parser.parse_args()

    dataset = args.dataset
    norm = np.inf if args.norm == 'inf' else int(args.norm)
    attack_type = args.attack_type

    config = template_config
    config['dataset']['dataset_name'] = dataset
    config['adv_samples']['norm_type'] = norm
    config['adv_samples']['search_mode'] = attack_type
    config['detector_adv_samples']['norm_type'] = norm
    config['detector_adv_samples']['search_mode'] = attack_type
    trees_model_v1 = full_experiment_cycle(config, load_from_disk=True)