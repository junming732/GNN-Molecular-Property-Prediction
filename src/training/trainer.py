import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


class MolecularTrainer:
    def __init__(self, model, optimizer, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_rmse": [],
            "val_rmse": [],
            "learning_rate": [],
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(batch)
            loss = F.mse_loss(pred, batch.y)
            mae = F.l1_loss(pred, batch.y)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_mae += mae.item()
            total_rmse += math.sqrt(loss.item())
            num_batches += 1

        if self.scheduler:
            self.scheduler.step()
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])

        return (
            total_loss / num_batches,
            total_mae / num_batches,
            total_rmse / num_batches,
        )

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                pred = self.model(batch)

                loss = F.mse_loss(pred, batch.y)
                mae = F.l1_loss(pred, batch.y)

                total_loss += loss.item()
                total_mae += mae.item()
                total_rmse += math.sqrt(loss.item())
                num_batches += 1

        return (
            total_loss / num_batches,
            total_mae / num_batches,
            total_rmse / num_batches,
        )

    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        best_val_mae = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_mae, train_rmse = self.train_epoch(train_loader)
            val_loss, val_mae, val_rmse = self.validate(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_mae"].append(train_mae)
            self.history["train_rmse"].append(train_rmse)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_rmse"].append(val_rmse)

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return self.history
