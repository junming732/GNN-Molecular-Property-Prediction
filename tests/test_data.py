from unittest.mock import patch

import pytest
import torch

from data.data_utils import batch_to_atoms, generate_otf_graph


class TestData:
    @patch("data.data_utils.radius_graph")
    def test_generate_otf_graph(self, mock_radius_graph):
        """Test graph generation from atomic positions"""
        # Mock the radius_graph function with correct shape
        # radius_graph returns a tensor of shape [2, num_edges]
        mock_radius_graph.return_value = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        )

        class MockData:
            def __init__(self):
                self.pos = torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],  # Distance 1 from origin
                        [2.0, 0.0, 0.0],  # Distance 2 from origin
                    ]
                )
                self.batch = torch.tensor([0, 0, 0])

        data = MockData()
        edge_index, edge_dist, distance_vec = generate_otf_graph(
            data, cutoff=1.5, max_neighbors=10, pbc=False
        )

        # Should create edges only between close atoms
        assert edge_index.shape[0] == 2  # Two rows for source and target
        assert edge_dist.numel() > 0  # Should have some edges
        assert mock_radius_graph.called

    def test_batch_to_atoms(self):
        """Test conversion of batch data to ASE atoms"""

        class MockData:
            def __init__(self):
                self.atomic_numbers = torch.tensor([1, 6, 8])  # H, C, O
                self.pos = torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
                self.batch = torch.tensor([0, 0, 0])

        data = MockData()
        atoms_list = batch_to_atoms(data)

        assert len(atoms_list) == 1  # Single molecule
        assert len(atoms_list[0]) == 3  # Three atoms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
