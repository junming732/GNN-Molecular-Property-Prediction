import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim


def hyperparameter_optimization(
    config_space, train_loader, val_loader, device, num_trials=50
):
    def objective(trial):
        # Suggest hyperparameters
        node_fea_dim = trial.suggest_categorical("node_fea_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        invariant = trial.suggest_categorical("invariant", [True, False])
        use_attention = trial.suggest_categorical("use_attention", [True, False])

        # Import here to avoid circular imports
        if use_attention:
            from models.attention_gnn import AttentionCGCNN as ModelClass
        else:
            from models.cgcnn import EnhancedCGCNN as ModelClass

        # Create model
        model = ModelClass(
            node_fea_dim=node_fea_dim,
            invariant=invariant,
            num_layers=num_layers,
            cutoff=12,
            max_neighbors=30,
            use_attention=use_attention,
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # Simple training loop for hyperparameter optimization
        best_val_mae = float("inf")

        for epoch in range(20):  # Reduced epochs for optimization
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_mae = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    val_mae += F.l1_loss(pred, batch.y).item()

            val_mae /= len(val_loader)
            if val_mae < best_val_mae:
                best_val_mae = val_mae

        return best_val_mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study
