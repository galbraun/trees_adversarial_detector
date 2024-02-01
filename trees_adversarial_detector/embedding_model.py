import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class EmbedModel(nn.Module):
    def __init__(self, num_samples, num_nodes, samples_embedding_dim=7, nodes_embedding_dim=50):
        super().__init__()

        self.node_embedding = nn.Embedding(num_nodes, nodes_embedding_dim)
        torch.nn.init.kaiming_uniform_(self.node_embedding.weight)

        self.samples_embedding = nn.Embedding(num_samples, samples_embedding_dim)
        torch.nn.init.kaiming_uniform_(self.samples_embedding.weight)

        # TODO: maybe add main_sample and context samples? and use different embedding for context?

        self.linear_1 = nn.Linear(2 * samples_embedding_dim + nodes_embedding_dim, 30)
        #self.linear_1 = nn.Linear(2 * samples_embedding_dim, 10)
        #torch.nn.init.kaiming_uniform_(self.linear_1.weight, nonlinearity='tanh')
        torch.nn.init.xavier_normal_(self.linear_1.weight)

        self.linear_2 = nn.Linear(30, 1)
        #torch.nn.init.kaiming_uniform_(self.linear_2.weight, nonlinearity='sigmoid')
        torch.nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, sample):
        sample_1_idx = torch.tensor(sample[:, 0]).to(torch.int64)
        sample_2_idx = torch.tensor(sample[:, 1]).to(torch.int64)
        node_idx = torch.tensor(sample[:, 2]).to(torch.int64)

        sample_1_embed = self.samples_embedding(sample_1_idx)
        sample_2_embed = self.samples_embedding(sample_2_idx)
        node_embed = self.node_embedding(node_idx)

        # TODO: try here different interactions between the embeddings
        concat_vec = torch.cat([sample_1_embed, sample_2_embed, node_embed], dim=-1)
        return torch.sigmoid(self.linear_2(torch.relu(self.linear_1(concat_vec))))

        # sample_1_node_interaction = torch.mul(sample_1_embed, node_embed)
        # sample_2_node_interaction = torch.mul(sample_2_embed, node_embed)
        # concat_vec = torch.cat([sample_1_node_interaction, sample_2_node_interaction], dim=-1)
        # return torch.sigmoid(self.linear_2(torch.relu(self.linear_1(concat_vec))))

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.y.shape[0]
