import logging
import pickle
import torch
import gc
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


class ImageDataset(Dataset):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        unique_labels = sorted(list(set(labels)))
        # for converting the labels into integers to use them in the dataloader
        self.id_label = {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = self.images[id]
        # only load the image when necessary (when we get the image)
        image = Image.open(image_path).convert("RGB")
        image = self._apply_transforms(image)
        label = self.labels[id]
        id_for_label = self.id_label[label]

        label = label = torch.tensor(id_for_label, dtype=torch.long)
        return {"img": image, "label": label}
    
    def _apply_transforms(self,image):
        return self.transforms(image)

class CNN(nn.Module):
    """
    ###########################################################################
    A CNN with two convolutional layers and three fully connected layers.
    All the images have been resized to 224x224 beforehand.
    ###########################################################################
    #############
    # Features: #
    #############
    # B x in_channels x 224 x 224
    # Number of kernels: K
    # Size of kernel: k x k
    # Stride: 1 x 1
    # Padding: same
    - Convolutional layer 1: 
        * in_channels = in_channels (defaults to RGB | in_channels = 3)
        * out_channels = 16
        * kernel_size = 3 x 3
        # Number of parameters conv1 = 3 x 3 x 3 x 16 = 432
        # Original input for next layer: B x 16 x 224 x 224
        # Input after MaxPool2d: B x 16 x 112 x 112
    - Convolutional layer 2:
        * in_channels = 16
        * out_channels = 32
        * kernel_size = 3 x 3
        # Number of parameters conv2 = 3 x 3 x 16 x 32 = 4608
        # Original input for next layer: B x 32 x 112 x 112
        # Input after MaxPool2d: B x 32 x 56 x 56
    - Convolutional layer 3:
        * in_channels = 32
        * out_channels = 64
        * kernel_size = 3 x 3
        # Number of parameters conv2 = 3 x 3 x 32 x 64 = 18432
        # Original input for next layer: B x 64 x 56 x 56
        # Input after MaxPool2d: B x 64 x 28 x 28
    ###########################################################################
        
    ###########################################################################
    ###############
    # Classifier: #
    ###############
    - Fully conected layer 1:



    ###########################################################################
    """

    def __init__(self, in_channels=3, n_labels=3):
        super(CNN, self).__init__()
        out_c = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c*2)

        self.conv3 = nn.Conv2d(in_channels=out_c*2, out_channels=out_c*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c*4)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(out_c*4 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Cache the data
all_hospital_data = None

def load_data(partition_id: int, batch_size: int):
    global all_hospital_data
    
    # Load the pickle file with the data that has been split by generate-data
    # beforehand.
    if all_hospital_data is None:
        with open("./data/processed/splits.pkl", "rb") as f:
            all_hospital_data = pickle.load(f)

    # Associate the partition_id with the corresponding hospital created with
    # the same id.
    hospital_name = f"hospital_{partition_id}"
    client_data = all_hospital_data[hospital_name]

    # Get a list of the images and labels
    def split(split_dict):
        images, labels = [], []
        for label, img_list in split_dict.items():
            images.extend(img_list)
            labels.extend([label] * len(img_list))
        return images, labels

    # Get the train and test images and labels
    train_imgs, train_labels = split(client_data["train"])
    test_imgs, test_labels = split(client_data["test"])

    # Create the local dataset after processing the images
    train_ds = ImageDataset(train_imgs, train_labels)
    test_ds = ImageDataset(test_imgs, test_labels)

    # Create dataloaders
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)

    gc.collect()

    return trainloader, testloader

def train_fedavg(model, trainloader, epochs, lr, device):
    model.to(device)
    # Use SGD with a CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test_fedavg(net, testloader, device):
    # Test the accuracy and loss
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def train_scaffold(global_c, local_c, model, trainloader, epochs, lr, device, **kwargs):
    model.to(device)
    # Use SGD with a CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss