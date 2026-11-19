import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNISTMLP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# Device
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Load test dataset
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================
# Load model
# ============================================================
model = MNISTMLP().to(device)
model.load_state_dict(torch.load("mnist_mlp.pth"))
model.eval()

# ============================================================
# Collect predictions, losses, labels
# ============================================================
all_preds = []
all_labels = []
all_losses = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='none')
        all_losses.extend(loss.cpu().numpy())
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_losses = np.array(all_losses)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================================
# Top 9 highest-loss examples
# ============================================================
top9_loss_idx = np.argsort(all_losses)[-9:]
plt.figure(figsize=(10,10))
for i, idx in enumerate(top9_loss_idx):
    image, label = test_dataset[idx]
    plt.subplot(3,3,i+1)
    plt.imshow(image[0], cmap='gray')
    plt.title(f"True: {label}, Pred: {all_preds[idx]}\nLoss: {all_losses[idx]:.2f}")
    plt.axis('off')
plt.suptitle("Top 9 highest-loss examples")
plt.show()

# ============================================================
# Confidently wrong examples
# ============================================================
confidences = []
wrong_indices = []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        for j in range(labels.size(0)):
            if preds[j] != labels[j]:
                wrong_indices.append(i*64 + j)
                confidences.append(probs[j, preds[j]].item())

top9_confident_wrong_idx = np.array(wrong_indices)[np.argsort(confidences)[-9:]]
plt.figure(figsize=(10,10))
for i, idx in enumerate(top9_confident_wrong_idx):
    image, label = test_dataset[idx]
    plt.subplot(3,3,i+1)
    plt.imshow(image[0], cmap='gray')
    plt.title(f"True: {label}, Pred: {all_preds[idx]}")
    plt.axis('off')
plt.suptitle("Top 9 confidently wrong examples")
plt.show()

# ============================================================
# Confusion Matrix
# ============================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
