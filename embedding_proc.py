import random

import keras
import numpy as np
from tqdm import tqdm
import math

from tree_models import extract_nodes_samples
from tree_models import extract_nodes_splits

K = keras.backend
KL = keras.layers

###########################################################################

def extract_embedding_dataset(X, model, embedding_dataset_size):
    forest_structure = extract_nodes_splits(model)

    nodes_samples = extract_nodes_samples(X, model)

    embedding_X, embedding_y, num_nodes = generate_dataset_routes(forest_structure, X, nodes_samples,
                                                                 num_of_samples=embedding_dataset_size)

    return embedding_X, embedding_y, num_nodes


def train_embedding(X_embedding, y_embedding, num_nodes, num_samples, num_features, val_size=0.1, epochs=5):
    # TODO: move this code to PyTorch
    input1 = KL.Input(shape=(1,))
    input2 = KL.Input(shape=(1,))

    nodeid = KL.Input(shape=(1,))

    samples_embedding = KL.Embedding(input_dim=num_samples, output_dim=math.floor(1.6*(num_features**0.55)), input_length=1)
    nodes_embedding = KL.Embedding(input_dim=num_nodes, output_dim=math.floor(1.6*(num_nodes**0.55)), input_length=1)


    encoded_s1 = samples_embedding(input1)
    encoded_s2 = samples_embedding(input2)
    encoded_node = nodes_embedding(nodeid)

    concat = KL.Concatenate(axis=-1)([encoded_s1, encoded_s2, encoded_node])
    concat_flatten = KL.Flatten()(concat)

    # change the network to be like this https://towardsdatascience.com/word2vec-made-easy-139a31a4b8ae
    num_hidden_neu = 2*math.floor(1.6 * (num_features ** 0.55)) + math.floor(1.6*(num_nodes**0.55))
    direction = KL.Dense(num_hidden_neu, activation='relu')(concat_flatten)
    #    direction = KL.Dense(20, activation=mish)(concat_flatten)
    direction = KL.Dense(1, activation='sigmoid')(direction)

    # direction = KL.Dense(1, activation='sigmoid')(concat_flatten)

    # f_model = keras.Model(inputs=[input1, input2, nodeid], outputs=direction)
    f_model = keras.Model(inputs=[input1, input2, nodeid], outputs=[direction])

    f_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    print(f_model.summary())

    train_indices = np.random.choice(X_embedding.shape[0], int((1 - val_size) * X_embedding.shape[0]))
    val_indices = np.arange(X_embedding.shape[0])[~np.isin(np.arange(X_embedding.shape[0]), train_indices)]

    training_generator = SamplesDataLoader(X_embedding[train_indices], y_embedding[train_indices])
    validation_genearator = SamplesDataLoader(X_embedding[val_indices], y_embedding[val_indices])

    history = f_model.fit_generator(generator=training_generator, validation_data=validation_genearator,
                                    use_multiprocessing=False, verbose=1, epochs=epochs)
    return history, f_model, f_model.summary()

class SamplesDataLoader(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=128):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.X))

    def generate_batch(self, samples_indexes):
        X = np.empty((self.batch_size, 3), dtype=float)
        y = np.empty(self.batch_size, dtype=float)

        for i, id in enumerate(samples_indexes):
            X[i, 0] = self.X[id][0]
            X[i, 1] = self.X[id][1]
            X[i, 2] = self.X[id][2]
            y[i] = self.y[id]


        perm_test = np.random.permutation(y.shape[0])
        X[:, 0] = X[perm_test, 0]
        X[:, 1] = X[perm_test, 1]
        X[:, 2] = X[perm_test, 2]
        y = y[perm_test]

        return X, y

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, batch_index):
        sample_indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        X, y = self.generate_batch(sample_indexes)
        return [X[:, 0], X[:, 1], X[:, 2]], y


def generate_dataset(split_nodes, X, num_of_samples=1000000):
    forest_nodes = split_nodes[['feature_index', 'Split']].values
    nodes_n = forest_nodes.shape[0]

    sampled_nodes = random.choices(range(nodes_n), k=num_of_samples)

    sampled_samples_1 = random.choices(range(X.shape[0]), k=num_of_samples)
    sampled_samples_2 = random.choices(range(X.shape[0]), k=num_of_samples)

    samples = []
    labels = []

    for i in tqdm(range(num_of_samples), total=num_of_samples):
        feature_index = int(forest_nodes[sampled_nodes[i]][0])
        split_value = float(forest_nodes[sampled_nodes[i]][1])

        sample_1_feat_value = X[sampled_samples_1[i]][feature_index]
        sample_2_feat_value = X[sampled_samples_2[i]][feature_index]

        if (sample_1_feat_value < split_value and sample_2_feat_value < split_value) or (
                sample_1_feat_value >= split_value and sample_2_feat_value >= split_value):
            same_direction = True
        else:
            same_direction = False

        labels.append(same_direction)
        samples.append((sampled_samples_1[i], sampled_samples_2[i], sampled_nodes[i]))

    return samples, labels, nodes_n


def generate_dataset_routes(split_nodes, X, nodes_samples, num_of_samples=1000000):
    forest_nodes_ids = split_nodes['ID'].values
    forest_nodes_feature = split_nodes['feature_index'].values
    forest_nodes_splits = split_nodes['Split'].values
    nodes_n = forest_nodes_ids.shape[0]

    sampled_nodes = []
    sampled_samples_1 = []
    sampled_samples_2 = []
    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Sample samples and nodes for embedding dataset'):
        sampled_nodes.append(random.choice(range(nodes_n)))

        sampled_samples_1.append(random.choice(nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))
        sampled_samples_2.append(random.choice(nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))

    samples = []
    context_samples = []
    labels = []

    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Calculating labels'):
        feature_index = int(forest_nodes_feature[sampled_nodes[i]])
        split_value = float(forest_nodes_splits[sampled_nodes[i]])

        sample_1_feat_value = X[sampled_samples_1[i]][feature_index]
        sample_2_feat_value = X[sampled_samples_2[i]][feature_index]

        if (sample_1_feat_value < split_value and sample_2_feat_value < split_value) or (
                sample_1_feat_value >= split_value and sample_2_feat_value >= split_value):
            same_direction = True
        else:
            same_direction = False
        # # TODO: maybe 4 classes? same_direction*2, different_direction*2
        labels.append(same_direction)
        samples.append((sampled_samples_1[i], sampled_samples_2[i], sampled_nodes[i]))

    return np.asarray(samples), np.asarray(labels), nodes_n