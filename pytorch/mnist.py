"""MNIST MLP training and evaluation with PyTorch.

Adapted from the author's learning exercise. Importing MNISTMLP does not train a model.
"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"

class MNISTMLP(nn.Module):
    """784 -> 128 -> 64 -> 10, with ReLU activations."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(self.flatten(x))

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    torch.manual_seed(42)
    device = get_device()
    print("Using device:", device)
    FIGURES.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=HERE / "data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=HERE / "data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, accuracies = [], []
    for epoch in range(5):
        model.train()
        total_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        epoch_loss, epoch_acc = total_loss / total, correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1}/5 | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    model.eval()
    test_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            test_loss += criterion(logits, labels).item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    test_acc = correct / total
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), HERE / "mnist_mlp.pth")
    print("Model saved to:", HERE / "mnist_mlp.pth")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, 6), losses, marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[1].plot(range(1, 6), accuracies, marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Training accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "pytorch_training_curves.png", dpi=160)
    plt.close(fig)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    wrong = (preds != labels).nonzero(as_tuple=False).flatten()[:9].tolist()
    if wrong:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, wrong):
            image, actual = test_dataset[index]
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"True: {actual}, Pred: {preds[index].item()}")
        fig.suptitle("Misclassified test images")
        fig.tight_layout()
        fig.savefig(FIGURES / "pytorch_misclassified.png", dpi=160)
        plt.close(fig)
    print("Figures saved to:", FIGURES)

if __name__ == "__main__":
    main()
