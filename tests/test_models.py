from unittest.mock import patch

import pytest
import torch

from models.attention_gnn import GraphAttentionLayer
from models.cgcnn import CGCNN, EnhancedCGCNN


class TestModels:
    def test_cgcnn_initialization(self):
        """Test that CGCNN model initializes correctly"""
        model = CGCNN(
            node_fea_dim=32, invariant=True, num_layers=3, cutoff=12.0, max_neighbors=30
        )
        assert model is not None
        assert hasattr(model, "embed")
        assert len(model.cgcnn_layers) == 3

    def test_enhanced_cgcnn_initialization(self):
        """Test that EnhancedCGCNN model initializes correctly"""
        model = EnhancedCGCNN(
            node_fea_dim=64,
            invariant=False,
            num_layers=4,
            cutoff=12.0,
            max_neighbors=30,
            use_attention=False,  # Set to False to avoid circular import
            dropout=0.1,
        )
        assert model is not None
        assert hasattr(model, "embed")
        assert len(model.cgcnn_layers) == 4

    def test_graph_attention_layer_initialization(self):
        """Test that GraphAttentionLayer initializes correctly"""
        layer = GraphAttentionLayer(
            in_channels=32, out_channels=32, edge_dim=3, heads=4, dropout=0.1
        )
        assert layer is not None
        assert hasattr(layer, "lin_key")
        assert hasattr(layer, "lin_query")
        assert hasattr(layer, "lin_value")

    @patch("models.cgcnn.generate_otf_graph")
    def test_model_forward_pass(self, mock_generate_otf_graph):
        """Test that models can perform forward pass without errors"""
        # Mock the generate_otf_graph function
        # edge_index should be shape [2, num_edges] - NOT [num_edges, 2]
        mock_edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 0]], dtype=torch.long
        )  # source nodes  # target nodes

        mock_edge_dist = torch.tensor([1.0, 1.0, 1.414], dtype=torch.float)
        mock_distance_vec = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=torch.float
        )

        mock_generate_otf_graph.return_value = (
            mock_edge_index,
            mock_edge_dist,
            mock_distance_vec,
        )

        # Create dummy data
        class MockData:
            def __init__(self):
                self.atomic_numbers = torch.tensor([1, 6, 8])  # H, C, O
                self.pos = torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
                self.batch = torch.tensor([0, 0, 0])

        data = MockData()

        # Test CGCNN
        model = CGCNN(
            node_fea_dim=16, invariant=True, num_layers=2, cutoff=12.0, max_neighbors=30
        )

        with torch.no_grad():
            output = model(data)
            assert output.shape == (1, 1)  # Batch size 1, output dimension 1
            assert mock_generate_otf_graph.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
