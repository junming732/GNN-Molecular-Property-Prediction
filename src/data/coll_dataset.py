import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm

class COLLDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['coll_data.xyz']  # Replace with actual data file

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download data if needed
        pass

    def process(self):
        # Read data into huge `Data` list
        data_list = []
        
        # Example: Process ASE atoms objects
        atoms_list = read(self.raw_paths[0], index=':')
        
        for i, atoms in enumerate(tqdm(atoms_list)):
            # Get node features (atomic numbers)
            atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            
            # Get positions
            pos = torch.tensor(atoms.positions, dtype=torch.float)
            
            # Get energy (target)
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            
            # Create graph data
            data = Data(
                atomic_numbers=atomic_numbers,
                pos=pos,
                y=energy,
                num_nodes=len(atomic_numbers)
            )
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
                
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])