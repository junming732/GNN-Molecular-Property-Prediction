from ase import Atoms
from torch_geometric.nn import radius_graph


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


def batch_to_atoms(data):
    """Convert batch of molecular data to ASE Atoms objects"""
    atoms_list = []
    if hasattr(data, "batch"):
        # Handle batched data
        batch_size = data.batch.max().item() + 1
        for i in range(batch_size):
            mask = data.batch == i
            atomic_numbers = data.atomic_numbers[mask].cpu().numpy()
            positions = data.pos[mask].cpu().numpy()
            atoms = Atoms(numbers=atomic_numbers, positions=positions)
            atoms_list.append(atoms)
    else:
        # Handle single molecule
        atomic_numbers = data.atomic_numbers.cpu().numpy()
        positions = data.pos.cpu().numpy()
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        atoms_list.append(atoms)

    return atoms_list
