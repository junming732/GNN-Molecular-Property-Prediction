import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add")

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(node_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # Propagate messages
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        m = self.bn(m)

        # Update node features
        out = self.update_net(torch.cat([x, m], dim=1))
        return out

    def message(self, x_i, x_j, edge_attr):
        # Create message
        input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(input)
