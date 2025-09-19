import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

# Add this function at the top of the file
def generate_otf_graph(data, cutoff, max_neighbors, pbc=False):
    """Generate graph on-the-fly from atomic positions"""
    from torch_geometric.nn import radius_graph
    
    edge_index = radius_graph(
        data.pos, 
        r=cutoff, 
        max_num_neighbors=max_neighbors,
        batch=data.batch if hasattr(data, 'batch') else None
    )
    
    # Calculate distances and vectors
    row, col = edge_index
    distance_vec = data.pos[row] - data.pos[col]
    edge_dist = distance_vec.norm(dim=-1)
    
    return edge_index, edge_dist, distance_vec

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.dropout = dropout
        self.out_channels = out_channels
        
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Edge feature processing
        self.lin_edge = nn.Linear(edge_dim, heads)
        
        self.lin_skip = nn.Linear(in_channels, heads * out_channels)
        self.bn = nn.BatchNorm1d(heads * out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_skip.weight)
        
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(edge_index, query=query, key=key, value=value, 
                            edge_attr=edge_attr)
        
        # Skip connection
        skip = self.lin_skip(x).view(-1, self.heads, self.out_channels)
        out = out + skip
        
        # Reshape and apply batch norm
        out = out.view(-1, self.heads * self.out_channels)
        out = self.bn(out)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_attr):
        # Compute attention weights
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # Add edge features to attention
        edge_attention = self.lin_edge(edge_attr).t()
        alpha = alpha + edge_attention
        
        # Apply softmax
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return value_j * alpha.unsqueeze(-1)

class AttentionCGCNN(nn.Module):
    def __init__(self, node_fea_dim, invariant, num_layers, cutoff, max_neighbors, heads=4, dropout=0.1):
        super().__init__()
        
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pbc = False
        
        if invariant:
            edge_fea_dim = 1
        else:
            edge_fea_dim = 3
        self.edge_fea_dim = edge_fea_dim

        self.embed = nn.Embedding(118, node_fea_dim)
        
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(node_fea_dim, node_fea_dim, edge_fea_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(node_fea_dim) for _ in range(num_layers)
        ])
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(dropout)
        
        self.pool = lambda x, batch: torch.mean(x, dim=0) if batch is None else torch.stack([
            torch.mean(x[batch == i], dim=0) for i in range(batch.max().item() + 1)
        ])
        
        self.out_mlp = nn.Sequential(
            nn.Linear(node_fea_dim, 2 * node_fea_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_fea_dim, 1)
        )

    def forward(self, data):
        edge_index, edge_dist, distance_vec = generate_otf_graph(
            data, self.cutoff, self.max_neighbors, self.pbc
        )
        
        if self.edge_fea_dim == 1:
            edge_attr = edge_dist.unsqueeze(-1)
        else:
            edge_attr = distance_vec

        h = self.embed(data.atomic_numbers)
        
        for i, layer in enumerate(self.attention_layers):
            h = layer(h, edge_index, edge_attr)
            h = self.bn_layers[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        
        # Global mean pooling
        if hasattr(data, 'batch'):
            h_pooled = self.pool(h, data.batch)
        else:
            h_pooled = self.pool(h, None)
            
        prediction = self.out_mlp(h_pooled)
        
        return prediction