import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from medmnist import BloodMNIST
from sklearn.metrics import confusion_matrix, classification_report

class CNN(nn.Module):

    def __init__(self, in_channels=3, n_labels=8):
        super(CNN, self).__init__()
        out_c = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c*2)

        self.conv3 = nn.Conv2d(in_channels=out_c*2, out_channels=out_c*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c*4)

        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((2,2))

        self.fc1 = nn.Linear(out_c*4 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():

    transforms = Compose([ToTensor(), 
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = BloodMNIST(split='train',root='./data/bloodmnist', download=True, transform=transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    validation_set = BloodMNIST(split='val',root='./data/bloodmnist', download=True, transform=transforms)
    val_loader = DataLoader(dataset=validation_set, batch_size=128, shuffle=True)
    test_set = BloodMNIST(split='test',root='./data/bloodmnist', download=True, transform=transforms)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True)

    model = CNN()

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