import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter

# Step 1: Data Preparation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

train_dir = '/content/Vegetable Images/train'
test_dir = '/content/Vegetable Images/test'
val_dir = '/content/Vegetable Images/validation'

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=data_transforms['val'])
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Step 2: Data Augmentation
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(),
    A.Normalize(),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(),
    ToTensorV2(),
])

# Step 3: Model Architecture
model = torchvision.models.resnet50(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Step 4: Training Setup
num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Step 5: Distributed Parallel Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = nn.DataParallel(model)

# Step 8: TensorBoard Logging
writer = SummaryWriter()

# Step 6: Training Loop
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)

    # Step 8: TensorBoard Logging
    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_acc, epoch)

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)

    # Step 8: TensorBoard Logging
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_acc, epoch)

    # Step 7: Validation
    print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # Save model checkpoint if validation accuracy improves
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_acc = val_acc

# Step 10: Testing and Deployment
# Load the best model checkpoint and use it for testing or deployment
best_model = torchvision.models.resnet50(pretrained=False)
best_model.fc = nn.Linear(best_model.fc.in_features, num_classes)
best_model = nn.DataParallel(best_model)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.to(device)
best_model.eval()
# ... (perform testing or deployment with the best model)

# Close the TensorBoard writer
writer.close()
