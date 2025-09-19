import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

# Add this function at the top of the file since it's used in multiple places
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

class CGCNNLayer(MessagePassing):
    def __init__(self, node_fea_dim, edge_fea_dim):
        super().__init__(aggr='add')
        self.node_fea_dim = node_fea_dim
        self.edge_fea_dim = edge_fea_dim
        
        # Message function parameters
        self.lin_f = nn.Linear(2 * node_fea_dim + edge_fea_dim, node_fea_dim)
        self.lin_s = nn.Linear(2 * node_fea_dim + edge_fea_dim, node_fea_dim)
        
        # Batch normalization
        self.bn_message = nn.BatchNorm1d(node_fea_dim)
        self.bn_update = nn.BatchNorm1d(node_fea_dim)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_f.weight)
        nn.init.xavier_uniform_(self.lin_s.weight)
        nn.init.zeros_(self.lin_f.bias)
        nn.init.zeros_(self.lin_s.bias)
        
    def forward(self, h, edge_index, edge_attr):
        # Propagate messages
        m = self.propagate(edge_index, h=h, e=edge_attr)
        m = self.bn_update(m)
        
        # Update node features
        out = h + m
        return out
        
    def message(self, h_i, h_j, e):
        # Concatenate features
        z = torch.cat([h_i, h_j, e], dim=1)
        
        # Compute message components
        m_f = self.lin_f(z)
        m_s = self.lin_s(z)
        
        # Apply activations
        m_f = self.sigmoid(m_f)
        m_s = self.softplus(m_s)
        
        # Element-wise multiplication
        m = m_f * m_s
        m = self.bn_message(m)
        
        return m

class CGCNN(nn.Module):
    def __init__(self, node_fea_dim, invariant, num_layers, cutoff, max_neighbors):
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
        
        self.cgcnn_layers = nn.ModuleList([
            CGCNNLayer(node_fea_dim, edge_fea_dim) for _ in range(num_layers)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(node_fea_dim) for _ in range(num_layers)
        ])
        self.activation = nn.Softplus()
        
        self.pool = lambda x, batch: torch.mean(x, dim=0) if batch is None else torch.stack([
            torch.mean(x[batch == i], dim=0) for i in range(batch.max().item() + 1)
        ])
        
        self.out_mlp = nn.Sequential(
            nn.Linear(node_fea_dim, 2 * node_fea_dim),
            nn.Softplus(),
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
        
        for i, layer in enumerate(self.cgcnn_layers):
            h = layer(h, edge_index, edge_attr)
            h = self.bn_layers[i](h)
            h = self.activation(h)
        
        # Global mean pooling
        if hasattr(data, 'batch'):
            h_pooled = self.pool(h, data.batch)
        else:
            h_pooled = self.pool(h, None)
            
        prediction = self.out_mlp(h_pooled)
        
        return prediction

class EnhancedCGCNN(nn.Module):
    def __init__(self, node_fea_dim, invariant, num_layers, cutoff, 
                 max_neighbors, use_attention=False, dropout=0.1):
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
        
        # Enhanced layers with optional attention
        if use_attention:
            from .attention_gnn import GraphAttentionLayer
            self.cgcnn_layers = nn.ModuleList([
                GraphAttentionLayer(node_fea_dim, node_fea_dim, edge_fea_dim) 
                for _ in range(num_layers)
            ])
        else:
            self.cgcnn_layers = nn.ModuleList([
                CGCNNLayer(node_fea_dim, edge_fea_dim) 
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
        
        # Enhanced output MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(node_fea_dim, 2 * node_fea_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_fea_dim, node_fea_dim),
            nn.Softplus(),
            nn.Linear(node_fea_dim, 1)
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
        
        for i, layer in enumerate(self.cgcnn_layers):
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