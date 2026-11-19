import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Make sure these exist: model, test_loader, test_dataset, device
model.eval()
losses = []
predictions = []
labels_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='none')

        losses.extend(loss.cpu().numpy())
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

losses = np.array(losses)
predictions = np.array(predictions)
labels_list = np.array(labels_list)

top9_loss_idx = np.argsort(losses)[-9:]

plt.figure(figsize=(10,10))
for i, idx in enumerate(top9_loss_idx):
    plt.subplot(3,3,i+1)
    plt.imshow(test_dataset[idx][0].squeeze(), cmap='gray')
    plt.title(f"True: {labels_list[idx]}, Pred: {predictions[idx]}\nLoss: {losses[idx]:.2f}")
    plt.axis('off')
plt.show()
