import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from medmnist import BloodMNIST
from sklearn.metrics import confusion_matrix, classification_report

class CNN(nn.Module):
    def __init__(self, in_channels=3, n_labels=8):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_labels)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():

    transformers = Compose([ToTensor(),Normalize(mean=[0.5], std=[0.5])])

    train_set = BloodMNIST(split='train',root='./data/bloodmnist', download=True, transform=transformers)
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    validation_set = BloodMNIST(split='val',root='./data/bloodmnist', download=True, transform=transformers)
    val_loader = DataLoader(dataset=validation_set, batch_size=128, shuffle=True)
    test_set = BloodMNIST(split='test',root='./data/bloodmnist', download=True, transform=transformers)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True)

    model = CNN(n_labels=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def validate(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets.squeeze()).sum().item()
        return 100 * correct / total

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze().long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_acc = validate(model, val_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Val Acc: {val_acc}%')

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()