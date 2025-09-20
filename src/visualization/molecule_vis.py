import matplotlib.pyplot as plt
import torch
from ase.visualize import plot

from ..data import batch_to_atoms


def visualize_molecule_with_predictions(molecule_data, model, original_model=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Visualize molecule
    atoms = batch_to_atoms(molecule_data)
    plot.plot_atoms(atoms[0], axes[0], rotation=("-75x, 45y, 10z"))
    axes[0].set_title("Molecular Structure")

    # Get predictions
    model.eval()
    with torch.no_grad():
        pred = model(molecule_data)
        true_energy = molecule_data.y.item()
        pred_energy = pred.item()

    if original_model:
        original_model.eval()
        with torch.no_grad():
            original_pred = original_model(molecule_data)
            original_pred_energy = original_pred.item()

    # Plot energy comparison
    models = ["Ground Truth", "Current Model"]
    energies = [true_energy, pred_energy]

    if original_model:
        models.append("Baseline Model")
        energies.append(original_pred_energy)

    axes[1].bar(models, energies)
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Energy Predictions")

    # Error analysis
    errors = [abs(pred_energy - true_energy)]
    if original_model:
        errors.append(abs(original_pred_energy - true_energy))
        models = ["Current Model", "Baseline Model"]
    else:
        models = ["Current Model"]

    axes[2].bar(models, errors)
    axes[2].set_ylabel("Absolute Error")
    axes[2].set_title("Prediction Errors")

    plt.tight_layout()
    plt.show()

    return pred_energy, true_energy


def plot_attention_weights(model, data, layer_idx=0):
    """Visualize attention weights for a given layer"""
    model.eval()
    with torch.no_grad():
        # Extract attention weights (implementation depends on model architecture)
        if hasattr(model.cgcnn_layers[layer_idx], "attention_weights"):
            attention_weights = model.cgcnn_layers[layer_idx].attention_weights
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weights.cpu().numpy(), cmap="viridis")
            plt.colorbar()
            plt.title(f"Attention Weights - Layer {layer_idx}")
            plt.xlabel("Head")
            plt.ylabel("Edge")
            plt.show()
