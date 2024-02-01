# import keras
import numpy as np
import random
from tqdm import tqdm

from trees_adversarial_detector.tree_models import extract_nodes_samples, extract_nodes_splits, \
    extract_nodes_splits_per_tree


# K = keras.backend
# KL = keras.layers


###########################################################################

def extract_embedding_dataset_pet_tree(X, model, embedding_dataset_size_per_tree, document_path=None):
    # Extract the structure of the ensemble model
    forest_structure = extract_nodes_splits_per_tree(model)

    # Calculate the distribution of the samples inside the model of trees
    nodes_samples = extract_nodes_samples(X, model)

    # TODO: split nodes_samples by the tree id, each node starts with "tree_id-node_id"
    nodes_samples_splited = {i: {n: nodes_samples[n] for n in nodes_samples if n.startswith(str(i))} for i in
                             range(len(model.get_booster().get_dump()))}

    # Generate a dataset for the embedding process
    dataset_per_tree = generate_dataset_routes_per_tree(forest_structure, X, nodes_samples_splited,
                                                        num_of_samples=embedding_dataset_size_per_tree)

    if document_path:
        print("Enter path so we can document everything")

    return dataset_per_tree


def extract_embedding_dataset(X, model, embedding_dataset_size, document_path=None):
    """
        Given a dataset and a trained model - extracting a dataset to calculate the embeddings of all of the samples
        in X regard the decision space in model

    :param X:
    :param model:
    :param embedding_dataset_size:
    :return:
    """

    # Extract the structure of the ensemble model
    forest_structure = extract_nodes_splits(model)

    # Calculate the distribution of the samples inside the model of trees
    nodes_samples = extract_nodes_samples(X, model)

    # Generate a dataset for the embedding process
    embedding_X, embedding_y, num_nodes = generate_dataset_routes(forest_structure, X, nodes_samples,
                                                                  num_of_samples=embedding_dataset_size)

    if document_path:
        print("Enter path so we can document everything")

    return embedding_X, embedding_y, num_nodes


def extract_new_samples_embedding_dataset(X_train, X_new, model, embedding_dataset_size, document_path=None):
    # Extract the structure of the ensemble model
    forest_structure = extract_nodes_splits(model)

    # Calculate the distribution of the samples inside the model of trees
    train_nodes_samples = extract_nodes_samples(X_train, model)
    new_nodes_samples = extract_nodes_samples(X_new, model)

    # Generate a dataset for the embedding process
    embedding_X, embedding_y, num_nodes = generate_dataset_routes_for_new_set(forest_structure, X_train, X_new,
                                                                              train_nodes_samples, new_nodes_samples,
                                                                              num_of_samples=embedding_dataset_size)

    if document_path:
        print("Enter path so we can document everything")

    return embedding_X, embedding_y, num_nodes


# def train_embedding(X_embedding, y_embedding, num_nodes, num_samples, num_features, val_size=0.1, epochs=5, logdir=None):
#     train_data = Data(X_embedding, y_embedding)
#     train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=2000, shuffle=True)
#
#     learning_rate = 0.05  # 3e-4
#     model = EmbedModel(num_nodes=num_nodes, num_samples=num_samples)
#     criterion = nn.BCELoss()
#     opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     scheduler = StepLR(opt, step_size=1, gamma=0.5)
#     epochs = 10
#
#     if logdir is None:
#         import socket
#         from datetime import datetime
#         current_time = datetime.now().strftime('%b%d_%H-%M-%S')
#         logdir = os.path.join(r'../runs', current_time + '_' + socket.gethostname())
#     writer = SummaryWriter(logdir)
#
#     i = 0
#
#     training_loss = 0.0
#     for epoch in range(epochs):
#         scheduler.step()
#         for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
#             model.train()
#             opt.zero_grad()
#             z = model(batch_x)
#             loss = criterion(z, batch_y.reshape(-1, 1))
#             loss.backward()
#             opt.step()
#
#             with torch.no_grad():
#                 model.eval()
#                 training_loss += loss.item()
#
#                 writer.add_scalar('training loss',
#                                   training_loss,
#                                   epoch * len(train_loader) + i)
#                 training_loss = 0.0
#     return model

import math
from tensorflow import keras

KL = keras.layers


def train_embedding(X_embedding, y_embedding, num_nodes, num_samples, num_features, val_size=0.1, epochs=5,
                    samples_embd_dim=None, node_embd_dim=None):
    # TODO: move this code to PyTorch
    input1 = KL.Input(shape=(1,))
    input2 = KL.Input(shape=(1,))

    nodeid = KL.Input(shape=(1,))

    smples_embd_dim_size = math.floor(1.6 * (num_features ** 0.55) if samples_embd_dim is None else samples_embd_dim)
    nodes_embd_dim_size = math.floor(1.6 * (num_nodes ** 0.55) if node_embd_dim is None else node_embd_dim)

    samples_embedding = KL.Embedding(input_dim=num_samples, output_dim=smples_embd_dim_size, input_length=1)
    nodes_embedding = KL.Embedding(input_dim=num_nodes, output_dim=nodes_embd_dim_size, input_length=1)

    encoded_s1 = samples_embedding(input1)
    encoded_s2 = samples_embedding(input2)
    encoded_node = nodes_embedding(nodeid)

    concat = KL.Concatenate(axis=-1)([encoded_s1, encoded_s2, encoded_node])
    concat_flatten = KL.Flatten()(concat)

    # change the network to be like this https://towardsdatascience.com/word2vec-made-easy-139a31a4b8ae
    num_hidden_neu = 2 * smples_embd_dim_size + nodes_embd_dim_size
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

    train_indices = np.random.choice(X_embedding.shape[0], int((1 - val_size) * X_embedding.shape[0]), replace=False)
    val_indices = np.arange(X_embedding.shape[0])[~np.isin(np.arange(X_embedding.shape[0]), train_indices)]

    training_generator = SamplesDataLoader(X_embedding[train_indices], y_embedding[train_indices])
    validation_genearator = SamplesDataLoader(X_embedding[val_indices], y_embedding[val_indices])

    history = f_model.fit_generator(generator=training_generator, validation_data=validation_genearator,
                                    use_multiprocessing=False, verbose=1, epochs=epochs)
    return history, f_model, f_model.summary()


def train_embedding_new_set(X_embedding, y_embedding, num_nodes, num_samples, num_features,
                            train_sample_embedding_weights, nodes_embedding_weights, hidden_layer_1_weights,
                            hidden_1_biases, hidden_layer_2_weights, hidden_2_biases, val_size=0.1, epochs=5,
                            samples_embd_dim=None, node_embd_dim=None):
    # TODO: move this code to PyTorch
    input1 = KL.Input(shape=(1,))
    input2 = KL.Input(shape=(1,))

    nodeid = KL.Input(shape=(1,))

    smples_embd_dim_size = math.floor(1.6 * (num_features ** 0.55) if samples_embd_dim is None else samples_embd_dim)
    nodes_embd_dim_size = math.floor(1.6 * (num_nodes ** 0.55) if node_embd_dim is None else node_embd_dim)

    new_samples_embedding = KL.Embedding(input_dim=num_samples, output_dim=smples_embd_dim_size, input_length=1)
    old_samples_embedding = KL.Embedding(input_dim=train_sample_embedding_weights.shape[0],
                                         output_dim=train_sample_embedding_weights.shape[1], input_length=1,
                                         weights=[train_sample_embedding_weights], trainable=False)

    nodes_embedding = KL.Embedding(input_dim=num_nodes, output_dim=nodes_embd_dim_size, input_length=1,
                                   weights=[nodes_embedding_weights], trainable=False)

    encoded_s1 = new_samples_embedding(input1)
    encoded_s2 = old_samples_embedding(input2)
    encoded_node = nodes_embedding(nodeid)

    concat = KL.Concatenate(axis=-1)([encoded_s1, encoded_s2, encoded_node])
    concat_flatten = KL.Flatten()(concat)

    # change the network to be like this https://towardsdatascience.com/word2vec-made-easy-139a31a4b8ae
    num_hidden_neu = 2 * smples_embd_dim_size + nodes_embd_dim_size
    hidden_layer_1 = KL.Dense(num_hidden_neu, activation='relu', weights=[hidden_layer_1_weights, hidden_1_biases],
                              trainable=False)

    direction = hidden_layer_1(concat_flatten)
    #    direction = KL.Dense(20, activation=mish)(concat_flatten)
    hidden_layer_2 = KL.Dense(1, activation='sigmoid', weights=[hidden_layer_2_weights, hidden_2_biases],
                              trainable=False)

    direction = hidden_layer_2(direction)

    # direction = KL.Dense(1, activation='sigmoid')(concat_flatten)

    # f_model = keras.Model(inputs=[input1, input2, nodeid], outputs=direction)
    f_model = keras.Model(inputs=[input1, input2, nodeid], outputs=[direction])

    f_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    print(f_model.summary())

    train_indices = np.random.choice(X_embedding.shape[0], int((1 - val_size) * X_embedding.shape[0]), replace=False)
    val_indices = np.arange(X_embedding.shape[0])[~np.isin(np.arange(X_embedding.shape[0]), train_indices)]

    training_generator = SamplesDataLoader(X_embedding[train_indices], y_embedding[train_indices])
    validation_genearator = SamplesDataLoader(X_embedding[val_indices], y_embedding[val_indices])

    history = f_model.fit_generator(generator=training_generator, validation_data=validation_genearator,
                                    use_multiprocessing=False, verbose=1, epochs=epochs)
    return history, f_model, f_model.summary()


class SamplesDataLoader(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=2048):
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


# def generate_dataset(split_nodes, X, num_of_samples=1000000):
#     forest_nodes = split_nodes[['feature_index', 'Split']].values
#     nodes_n = forest_nodes.shape[0]
#
#     sampled_nodes = random.choices(range(nodes_n), k=num_of_samples)
#
#     sampled_samples_1 = random.choices(range(X.shape[0]), k=num_of_samples)
#     sampled_samples_2 = random.choices(range(X.shape[0]), k=num_of_samples)
#
#     samples = []
#     labels = []
#
#     for i in tqdm(range(num_of_samples), total=num_of_samples):
#         feature_index = int(forest_nodes[sampled_nodes[i]][0])
#         split_value = float(forest_nodes[sampled_nodes[i]][1])
#
#         sample_1_feat_value = X[sampled_samples_1[i]][feature_index]
#         sample_2_feat_value = X[sampled_samples_2[i]][feature_index]
#
#         if (sample_1_feat_value < split_value and sample_2_feat_value < split_value) or (
#                 sample_1_feat_value >= split_value and sample_2_feat_value >= split_value):
#             same_direction = True
#         else:
#             same_direction = False
#
#         labels.append(same_direction)
#         samples.append((sampled_samples_1[i], sampled_samples_2[i], sampled_nodes[i]))
#
#     return samples, labels, nodes_n


# def generate_dateset_routes_full(split_nodes, X, nodes_samples, max_from_each_node=100000):
#     relevant_nodes = split_nodes['ID'].to_list()
#
#     sampled_nodes = []
#     sampled_samples_1 = []
#     sampled_samples_2 = []
#
#     for i, node in tqdm(enumerate(relevant_nodes), total=len(relevant_nodes), desc='Generating triplets'):
#         combs = random.sample(list(combinations(nodes_samples[node], 2)), k=max_from_each_node)
#         first = [x[0] for x in combs]
#         second = [x[0] for x in combs]
#
#         sampled_nodes += [i]*len(first)
#         sampled_samples_1.append(first)
#         sampled_samples_2.append(second)


import pandas as pd


def generate_dataset_routes_per_tree(split_nodes, X, nodes_samples, num_of_samples=1000000):
    dataset_per_tree = {i: None for i in split_nodes.keys()}
    aggregated_split_nodes = pd.DataFrame()
    aggregated_nodes_samples = {}
    for i in tqdm(split_nodes.keys(), total=len(split_nodes.keys()), desc="Extract dataset per tree"):
        aggregated_split_nodes = pd.concat([aggregated_split_nodes, split_nodes[i]])
        aggregated_nodes_samples = {**aggregated_nodes_samples, **nodes_samples[i]}
        dataset_per_tree[i] = generate_dataset_routes(aggregated_split_nodes, X, aggregated_nodes_samples,
                                                      num_of_samples,
                                                      disable_tqdm=True)
    return dataset_per_tree


def generate_dataset_routes(split_nodes, X, nodes_samples, num_of_samples=1000000, disable_tqdm=False):
    """
        Creating a dataset for the embedding process.
        <node, sample_i, sample_j, does the samples agree on the node split? >

        #TODO: change this dataset generation to a pytorch datagenerator?
    :param split_nodes:
    :param X:
    :param nodes_samples:
    :param num_of_samples:
    :return:
    """
    forest_nodes_ids = split_nodes['ID'].values
    # forest_nodes_feature = split_nodes['feature_index'].values
    forest_nodes_feature = split_nodes['feature_index'].values
    forest_nodes_splits = split_nodes['Split'].values
    nodes_n = forest_nodes_ids.shape[0]

    sampled_nodes = []
    sampled_samples_1 = []
    sampled_samples_2 = []
    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Sample samples and nodes for embedding dataset',
                  disable=disable_tqdm):
        sampled_nodes.append(random.choice(range(nodes_n)))

        sampled_samples_1.append(random.choice(nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))
        sampled_samples_2.append(random.choice(nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))

    samples = []
    context_samples = []
    labels = []

    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Calculating labels', disable=disable_tqdm):
        # feature_index = int(forest_nodes_feature[sampled_nodes[i]])
        feature = forest_nodes_feature[sampled_nodes[i]]
        split_value = float(forest_nodes_splits[sampled_nodes[i]])

        sample_1_feat_value = X[sampled_samples_1[i], feature]
        sample_2_feat_value = X[sampled_samples_2[i], feature]

        if (sample_1_feat_value < split_value and sample_2_feat_value < split_value) or (
                sample_1_feat_value >= split_value and sample_2_feat_value >= split_value):
            same_direction = True
        else:
            same_direction = False
        # # TODO: maybe 4 classes? same_direction*2, different_direction*2
        labels.append(same_direction)
        samples.append((sampled_samples_1[i], sampled_samples_2[i], sampled_nodes[i]))

    return np.asarray(samples), np.asarray(labels), nodes_n


def generate_dataset_routes_for_new_set(split_nodes, X_train, X_new,
                                        train_nodes_samples, new_nodes_samples,
                                        num_of_samples=1000000, disable_tqdm=False):
    """
        Creating a dataset for the embedding process.
        <node, sample_i, sample_j, does the samples agree on the node split? >

        #TODO: change this dataset generation to a pytorch datagenerator?
    :param split_nodes:
    :param X:
    :param nodes_samples:
    :param num_of_samples:
    :return:
    """

    relevant_ids = [l_id for l_id, n_idxs in new_nodes_samples.items() if len(n_idxs) != 0]
    relevnat_splits = split_nodes[split_nodes['ID'].isin(relevant_ids)]
    forest_nodes_ids = relevnat_splits['ID'].values
    forest_nodes_feature = relevnat_splits['feature_index'].values
    forest_nodes_splits = relevnat_splits['Split'].values
    nodes_n = relevnat_splits.shape[0]

    sampled_nodes = []
    sampled_samples_1 = []
    sampled_samples_2 = []
    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Sample samples and nodes for embedding dataset',
                  disable=disable_tqdm):
        sampled_nodes.append(random.choice(range(nodes_n)))

        sampled_samples_1.append(random.choice(new_nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))
        sampled_samples_2.append(random.choice(train_nodes_samples[forest_nodes_ids[sampled_nodes[-1]]]))

    samples = []
    context_samples = []
    labels = []

    for i in tqdm(range(num_of_samples), total=num_of_samples, desc='Calculating labels', disable=disable_tqdm):
        # feature_index = int(forest_nodes_feature[sampled_nodes[i]])
        feature = forest_nodes_feature[sampled_nodes[i]]
        split_value = float(forest_nodes_splits[sampled_nodes[i]])

        sample_1_feat_value = X_new[sampled_samples_1[i], feature]
        sample_2_feat_value = X_train[sampled_samples_2[i], feature]

        if (sample_1_feat_value < split_value and sample_2_feat_value < split_value) or (
                sample_1_feat_value >= split_value and sample_2_feat_value >= split_value):
            same_direction = True
        else:
            same_direction = False
        # # TODO: maybe 4 classes? same_direction*2, different_direction*2
        labels.append(same_direction)
        samples.append((sampled_samples_1[i], sampled_samples_2[i], sampled_nodes[i]))

    return np.asarray(samples), np.asarray(labels), nodes_n
