"""Standalone error analysis for a previously trained MNISTMLP."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist import MNISTMLP, get_device

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"

def main():
    device = torch.device("cpu")
    weights = HERE / "mnist_mlp.pth"
    if not weights.exists():
        raise FileNotFoundError(f"No trained model at {weights}; run 'python pytorch/mnist.py' first")
    FIGURES.mkdir(parents=True, exist_ok=True)
    test_dataset = datasets.MNIST(root=HERE / "data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = MNISTMLP().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    preds_list, labels_list, losses_list, confidences_list = [], [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses_list.extend(F.cross_entropy(logits, labels, reduction="none").cpu().numpy())
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            preds_list.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            confidences_list.extend(probs.gather(1, pred.unsqueeze(1)).flatten().cpu().numpy())
    preds = np.asarray(preds_list)
    labels = np.asarray(labels_list)
    losses = np.asarray(losses_list)
    confidences = np.asarray(confidences_list)
    print(f"Test accuracy: {(preds == labels).mean():.4f}")

    def plot_examples(indices, title, filename):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, indices):
            image, true_label = test_dataset[int(index)]
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"True {true_label}, pred {preds[index]}\nLoss {losses[index]:.2f}")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=160)
        plt.close(fig)

    plot_examples(np.argsort(losses)[-9:][::-1], "Highest-loss test examples", "pytorch_highest_loss.png")
    wrong = np.flatnonzero(preds != labels)
    if len(wrong):
        ranked_wrong = wrong[np.argsort(confidences[wrong])[-9:][::-1]]
        plot_examples(ranked_wrong, "Most confidently misclassified examples", "pytorch_confident_errors.png")

    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="MNIST test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "pytorch_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("Analysis figures saved to:", FIGURES)

if __name__ == "__main__":
    main()
