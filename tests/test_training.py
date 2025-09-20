import pytest
import torch

from models.cgcnn import CGCNN
from training.trainer import MolecularTrainer


class TestTraining:
    def test_molecular_trainer_initialization(self):
        """Test that MolecularTrainer initializes correctly"""
        model = CGCNN(
            node_fea_dim=32, invariant=True, num_layers=2, cutoff=12.0, max_neighbors=30
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = MolecularTrainer(model, optimizer, device="cpu")

        assert trainer is not None
        assert trainer.device == "cpu"
        assert hasattr(trainer, "history")

    def test_trainer_history(self):
        """Test that trainer maintains history correctly"""
        model = CGCNN(
            node_fea_dim=16, invariant=True, num_layers=1, cutoff=12.0, max_neighbors=30
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = MolecularTrainer(model, optimizer, device="cpu")

        # Simulate adding to history
        trainer.history["train_loss"].append(0.5)
        trainer.history["val_loss"].append(0.4)

        assert len(trainer.history["train_loss"]) == 1
        assert len(trainer.history["val_loss"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
