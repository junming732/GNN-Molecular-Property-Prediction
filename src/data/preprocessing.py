from torch_geometric.nn import radius_graph
from torch_geometric.transforms import BaseTransform


def generate_otf_graph(data, cutoff, max_neighbors, pbc=False):
    """Generate graph on-the-fly from atomic positions"""
    edge_index = radius_graph(
        data.pos,
        r=cutoff,
        max_num_neighbors=max_neighbors,
        batch=data.batch if hasattr(data, "batch") else None,
    )

    # Calculate distances and vectors
    row, col = edge_index
    distance_vec = data.pos[row] - data.pos[col]
    edge_dist = distance_vec.norm(dim=-1)

    return edge_index, edge_dist, distance_vec


class MolecularPreprocessor(BaseTransform):
    def __init__(self, cutoff=12.0, max_neighbors=30):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def __call__(self, data):
        # Generate graph structure
        edge_index, edge_dist, distance_vec = generate_otf_graph(data, self.cutoff, self.max_neighbors)

        data.edge_index = edge_index
        data.edge_dist = edge_dist
        data.distance_vec = distance_vec

        return data
