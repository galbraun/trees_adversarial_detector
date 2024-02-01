import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LeafsSamplesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LeafsSamplesDataModule(pl.LightningDataModule):
    def __init__(self, tree_model, tree_index, X, n_samples, batch_size):
        super().__init__()
        self.tree_model = tree_model
        self.batch_size = batch_size
        self.X = X
        self.tree_index = tree_index
        self.n_samples = n_samples

    def setup(self, stage=None):
        trees_df = self.tree_model.get_booster().trees_to_dataframe()
        tree_df = trees_df[trees_df['Tree'] == self.tree_index]
        leaf_unique_values = tree_df[tree_df['Feature'] == 'Leaf'].Node.values
        # leaf_idx_reindexed = dict(zip(leaf_unique_values, range(len(leaf_unique_values))))

        leaf_to_samples = {leaf_val: np.where(self.tree_model.apply(self.X)[:, self.tree_index] == leaf_val)[0] for
                           leaf_val in
                           leaf_unique_values}

        samples_X = []
        samples_y = []

        # sample samples from different leafs
        diff_leaf_pairs = [np.random.choice(leaf_unique_values, size=2, replace=False) for i in
                           range(self.n_samples // 2)]
        for leaf_pair in diff_leaf_pairs:
            samples_X.append(
                (np.random.choice(leaf_to_samples[leaf_pair[0]]), np.random.choice(leaf_to_samples[leaf_pair[1]])))
            samples_y.append(0)

        # sample samples from same leaf
        same_leaf_pairs = np.random.choice(leaf_unique_values, size=self.n_samples // 2)
        for leaf_val in same_leaf_pairs:
            samples_X.append((np.random.choice(leaf_to_samples[leaf_val]), np.random.choice(leaf_to_samples[leaf_val])))
            samples_y.append(1)

        samples_X = np.array(samples_X)
        samples_y = np.array(samples_y, dtype=np.float32).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(samples_X, samples_y, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        self.train_samples = LeafsSamplesDataset(X_train, y_train)
        self.val_samples = LeafsSamplesDataset(X_val, y_val)
        self.test_samples = LeafsSamplesDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_samples, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_samples, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_samples, batch_size=self.batch_size)


class LeafsNet(pl.LightningModule):
    def __init__(self, num_samples, embedding_dim):
        super(LeafsNet, self).__init__()

        self.samples_embedding = nn.Embedding(num_samples, embedding_dim)

        self.linear_cat = nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim)
        self.hidden_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        sample_1_idx = x[:, 0]
        sample_2_idx = x[:, 1]

        sample_1_embd = self.samples_embedding(sample_1_idx)
        sample_2_embd = self.samples_embedding(sample_2_idx)

        # interaction_vec = torch.mul(sample_1_embd, sample_2_embd)
        # hidden_vec = torch.relu(self.hidden_linear(interaction_vec))
        #
        # return torch.sigmoid(self.linear(hidden_vec))

        cat_vec = torch.cat([sample_1_embd, sample_2_embd], dim=1)
        cat_linear = torch.relu(self.linear_cat(cat_vec))
        hidden_vec = torch.relu(self.hidden_linear(cat_linear))

        return torch.sigmoid(self.linear(hidden_vec))

    def training_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)

        correct = (y_hat > 0.5).int().eq(y).sum().item()
        total = len(y)

        logs = {"train_loss": loss}
        output = {
            "loss": loss,
            "log": logs,
            "correct": correct,
            "total": total
        }
        return output

    def validation_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)

        correct = (y_hat > 0.5).int().eq(y).sum().item()
        total = len(y)

        logs = {"test_loss": loss}
        output = {
            "loss": loss,
            "log": logs,
            "correct": correct,
            "total": total
        }
        return output

    def test_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.BCELoss()(y_hat, y)

        correct = (y_hat > 0.5).int().eq(y).sum().item()
        total = len(y)

        logs = {"test_loss": loss}
        output = {
            "loss": loss,
            "log": logs,
            "correct": correct,
            "total": total
        }
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'loss': avg_loss, "accuracy": correct / total}
        epoch_dictionary = {
            'loss': avg_loss,
            'log': tensorboard_logs}
        return epoch_dictionary

        # train_idxs_all_leafs = []
        # val_idxs_all_leafs = []
        # test_idxs_all_leafs = []
        #
        # for leaf_val in leaf_unique_values:
        #     leaf_samples_indexes = np.where(self.tree_model.apply(self.X)[:, 0] == leaf_val)
        #     train_idxs, val_idxs, test_idxs = np.split(np.random.permutation(leaf_samples_indexes),
        #                                                [int(len(leaf_samples_indexes) * 0.5),
        #                                                 int(len(leaf_samples_indexes) * 0.75)])
        #     train_idxs_all_leafs.append(train_idxs)
        #     val_idxs_all_leafs.append(val_idxs)
        #     test_idxs_all_leafs.append(test_idxs)
        #
        # train_idxs_all_leafs = np.hstack(train_idxs_all_leafs)
        # val_idxs_all_leafs = np.hstack(val_idxs_all_leafs)
        # test_idxs_all_leafs = np.hstack(test_idxs_all_leafs)
