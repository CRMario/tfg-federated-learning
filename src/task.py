import logging
import pickle
import copy
import torch
import gc
import json
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import Array, ArrayRecord
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.metrics import confusion_matrix, precision_score


class ImageDataset(Dataset):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        with open('./data/processed/label_mappings.json', "r") as f:
            label_mappings = json.load(f)

        self.id_label = label_mappings["id_to_label"]
        self.label_id = label_mappings["label_to_id"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = self.images[id]
        # only load the image when necessary (when we get the image)
        image = Image.open(image_path).convert("RGB")
        image = self._apply_transforms(image)
        label = self.labels[id]
        id_for_label = self.label_id[label]

        label = torch.tensor(id_for_label, dtype=torch.long)
        return {"img": image, "label": label}
    
    def _apply_transforms(self,image):
        return self.transforms(image)
    
    def get_label_name_map(self):
        return self.id_label
    
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

    return trainloader, testloader, test_ds.get_label_name_map()

def train_fedavg(model, trainloader, epochs, lr, device, **kwargs):
    model.to(device)
    # Use SGD with a CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # get training accuracy
            _, predicted = torch.max(out,dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
    avg_trainloss = running_loss / len(trainloader)
    train_accuracy = correct / total
    return avg_trainloss, train_accuracy, {}, {}, {}


def test(net, testloader, device, all_labels):
    # Test the accuracy and loss
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    test_predictions, test_labels = [], []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    accuracy = 0.0
    f_loss = 0.0
    if len(testloader.dataset) > 0:
        accuracy = correct / len(testloader.dataset)
        f_loss = loss / len(testloader)
        cm = confusion_matrix(y_true=test_labels,
                            y_pred=test_predictions,
                            labels=all_labels)
        precision = precision_score(y_true=test_labels, 
                                    y_pred=test_predictions, 
                                    labels=all_labels, 
                                    average='macro', 
                                    zero_division=0
        )
    return f_loss, accuracy, cm.flatten().tolist(), precision


def train_scaffold(global_c, local_c, model, trainloader, epochs, lr, device, **kwargs):

    model_params = model.state_dict()
    ini_model = copy.deepcopy(model_params)
    model.to(device)
    # Use SGD with a CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    steps = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()

            # disable grad to modify the grad with glocal and local control variables
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        c = torch.from_numpy(global_c[name].numpy()).to(device)
                        c_k = torch.from_numpy(local_c[name].numpy()).to(device)
                        param.grad.data -= c_k
                        param.grad.data += c 

            optimizer.step()
            running_loss += loss.item()
            steps += 1
            # get training accuracy
            _, predicted = torch.max(out,dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

    training_accuracy = correct / total
    avg_trainloss = running_loss / len(trainloader)

    c_weight = 1 / (steps * lr)

    next_c, w_diff, c_diff = {}, {}, {}

    for name in ini_model.keys():
        c_k = torch.from_numpy(local_c[name].numpy()).to(device)
        c = torch.from_numpy(global_c[name].numpy()).to(device)
        w_ini = ini_model[name].to(device)
        w = model_params[name].to(device)
        next_c_value = (c_k - c + (c_weight * (w_ini - w)))
        next_c[name] = next_c_value.detach().cpu().numpy()
        w_diff[name] = (w - w_ini).detach().cpu()
        c_diff[name] = (next_c_value - c_k).detach().cpu()

    return avg_trainloss, training_accuracy, w_diff, c_diff, ArrayRecord(array_dict={k: Array(v) for k,v in next_c.items()})

def train_fedprox(proximal_mu, inexact_threshold, model, trainloader, epochs, lr, device, **kwargs):
   
    global_model_params = [parameter.clone() for parameter in model.parameters()]
    model.to(device)
    # Use SGD with a CrossEntropyLoss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Calculate the inexact condition
    model.train()
    first_batch = next(iter(trainloader))
    optimizer.zero_grad()
    init_loss = criterion(model(first_batch["img"].to(device)), first_batch["label"].to(device))
    init_loss.backward()

    # Get the initial norm
    with torch.no_grad():
        initial_grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm(2)

    optimizer.zero_grad()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            diff = 0.0
            for local_weight, global_weight in zip(model.parameters(), global_model_params):
                diff += (local_weight - global_weight).norm(2)**2
            proximal_term = (proximal_mu / 2) * diff
            loss = loss + proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(out,dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        
        with torch.no_grad():
            current_grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm(2)

        if current_grad_norm <= initial_grad_norm * inexact_threshold:
            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss, {}, {}, {}
        
    avg_trainloss = running_loss / len(trainloader)
    training_accuracy = correct / total
    return avg_trainloss, training_accuracy, {}, {}, {}

def aggregate_malicious_vector(model):
    # For now unimplemented
    pass