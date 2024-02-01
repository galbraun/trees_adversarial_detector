import faiss
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from trees_adversarial_detector.embedding_proc import train_embedding, extract_embedding_dataset_per_tree, \
    extract_new_samples_embedding_dataset_per_tree, train_embedding_new_set
from trees_adversarial_detector.evaluation import FaissKNeighbors


def calculate_a_priori_class_switching(label_sequences):
    probabilities = []
    for i in range(1, label_sequences.shape[1]):
        probabilities.append(
            np.count_nonzero(label_sequences[:, i] != label_sequences[:, i - 1]) / label_sequences.shape[0])
    return probabilities


def calculate_sample_switching_bayesian_log_likelihood(sample_labels_sequence, probabilities):
    sample_log_likelihood = 0
    for i in range(1, len(sample_labels_sequence)):
        if sample_labels_sequence[i] != sample_labels_sequence[i - 1]:
            sample_log_likelihood += np.log(probabilities[i - 1])
        else:
            sample_log_likelihood += np.log(1 - probabilities[i - 1])
    return sample_log_likelihood


def train_single_round_classifier(X_embedded, y, dimensions=100):
    pca_projector = PCA(n_components=min([dimensions, X_embedded.shape[1]])).fit(X_embedded, y)
    projected_X = pca_projector.transform(X_embedded)

    round_classifier = FaissKNeighbors(k=5, faiss=faiss)
    round_classifier.fit(projected_X.astype('float32'), y.astype('float32'))
    return round_classifier, pca_projector


def train_rounds_classifiers(train_embedding_models, y, n_estimators):
    round_classifiers = []
    for i in range(n_estimators):
        train_samples_embedding = train_embedding_models[i].layers[3].get_weights()[0]
        round_classifiers.append(train_single_round_classifier(train_samples_embedding, y))
    return round_classifiers


def generate_embedding_by_rounds(X, model, embedding_dataset_size_per_tree, n_classes, samples_embd_dim, node_embd_dim):
    dataset_per_tree = extract_embedding_dataset_per_tree(X, model, embedding_dataset_size_per_tree, n_classes)

    histories = []
    embedding_models = []
    summaries = []
    for i in range(model.n_estimators):
        history_tree_i, embedding_model_tree_i, summary_tree_i = train_embedding(dataset_per_tree[i][0],
                                                                                 dataset_per_tree[i][1],
                                                                                 dataset_per_tree[i][2], X.shape[0],
                                                                                 X.shape[1], epochs=1,
                                                                                 samples_embd_dim=samples_embd_dim,
                                                                                 node_embd_dim=node_embd_dim)
        histories.append(history_tree_i)
        embedding_models.append(embedding_model_tree_i)
        summaries.append(summary_tree_i)

    return histories, embedding_models, summaries


def generate_embedding_by_round_new_samples(X_train, X_new, model, embedding_dataset_size_per_tree,
                                            train_embedding_models, n_classes, samples_embd_dim, node_embd_dim):
    dataset_per_tree = extract_new_samples_embedding_dataset_per_tree(X_train, X_new, model,
                                                                      embedding_dataset_size_per_tree, n_classes)

    histories = []
    embedding_models = []
    summaries = []
    for i in range(model.n_estimators):
        train_samples_embedding = train_embedding_models[i].layers[3].get_weights()[0]
        train_nodes_embedding = train_embedding_models[i].layers[4].get_weights()[0]
        hidden_1_weights = train_embedding_models[i].layers[7].get_weights()[0]
        hidden_1_biases = train_embedding_models[i].layers[7].get_weights()[1]
        hidden_2_weights = train_embedding_models[i].layers[8].get_weights()[0]
        hidden_2_biases = train_embedding_models[i].layers[8].get_weights()[1]

        history_tree_i, embedding_model_tree_i, summary_tree_i = train_embedding_new_set(dataset_per_tree[i][0],
                                                                                         dataset_per_tree[i][1],
                                                                                         dataset_per_tree[i][2],
                                                                                         X_new.shape[0],
                                                                                         X_new.shape[1],
                                                                                         train_samples_embedding,
                                                                                         train_nodes_embedding,
                                                                                         hidden_1_weights,
                                                                                         hidden_1_biases,
                                                                                         hidden_2_weights,
                                                                                         hidden_2_biases, epochs=1,
                                                                                         samples_embd_dim=samples_embd_dim,
                                                                                         node_embd_dim=node_embd_dim)

        histories.append(history_tree_i)
        embedding_models.append(embedding_model_tree_i)
        summaries.append(summary_tree_i)

    return histories, embedding_models, summaries


def calculate_single_round_embeddings(X):
    pass
