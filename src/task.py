import pickle
import copy
import torch
import gc
import math
import numpy as np
from flwr.app import Array, ArrayRecord, MetricRecord
from src.image_dataset import IMAGE_DATASET, ImageDatasetLocal
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score
from utils.config import load_config
from src.model import MODEL, CNN_Local

# Cache the data
all_client_data = None

def load_data(partition_id: int, batch_size: int):
    global all_client_data
    
    # Load the pickle file with the data that has been split by generate-data
    # beforehand.
    if all_client_data is None:
        with open("./data/processed/splits.pkl", "rb") as f:
            all_client_data = pickle.load(f)
    
    # Associate the partition_id with the corresponding client created with
    # the same id.
    client_name = f"client_{partition_id}"
    client_data = all_client_data[client_name]

    train_imgs = client_data["train"]["x"]
    train_labels = client_data["train"]["y"]
    test_imgs = client_data["test"]["x"]
    test_labels = client_data["test"]["y"]

    dataset = load_config("./data/processed/config.json")["dataset"]
    image_dataset_builder = IMAGE_DATASET.get(dataset,ImageDatasetLocal)

    # Create the local dataset after processing the images
    train_ds = image_dataset_builder(train_imgs, train_labels)
    test_ds = image_dataset_builder(test_imgs, test_labels)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)

    gc.collect()

    return trainloader, testloader

def load_centralised_dataset():
    global all_client_data
    
    if all_client_data is None:
        with open("./data/processed/splits.pkl", "rb") as f:
            all_client_data = pickle.load(f)
    
    all_test_x = []
    all_test_y = []

    for client_name in all_client_data:
        all_test_x.append(all_client_data[client_name]["test"]["x"])
        all_test_y.append(all_client_data[client_name]["test"]["y"])

    test_imgs = np.concatenate(all_test_x, axis=0)
    test_labels = np.concatenate(all_test_y, axis=0)

    dataset = load_config("./data/processed/config.json")["dataset"]
    image_dataset_builder = IMAGE_DATASET.get(dataset,ImageDatasetLocal)
    
    test_ds = image_dataset_builder(test_imgs, test_labels)

    testloader = DataLoader(test_ds, shuffle=False)

    gc.collect()

    return testloader

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
    avg_trainloss = running_loss / (epochs * len(trainloader))
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
    avg_trainloss = running_loss / (epochs * len(trainloader))

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
    #with torch.no_grad():
    #    initial_grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm(2)

    optimizer.zero_grad()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    current_epoch = 1
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
            modified_loss = loss + proximal_term

            modified_loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(out,dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        
        #with torch.no_grad():
        #    current_grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm(2)

        
        #if current_grad_norm <= initial_grad_norm * inexact_threshold:
        #    avg_trainloss = running_loss / (current_epoch * len(trainloader))
        #    training_accuracy = correct / total
        #    return avg_trainloss, training_accuracy, {}, {}, {}
        # TODO: fix the "stragglers" part of the algorithm. 
        current_epoch += 1
        
    avg_trainloss = running_loss / (epochs * len(trainloader))
    training_accuracy = correct / total
    return avg_trainloss, training_accuracy, {}, {}, {}

def aggregate_malicious_vector(model):
    # For now unimplemented
    pass

def global_evaluate(server_round: int, arrays: ArrayRecord):

    config = load_config("./data/processed/config.json")
    mappings = load_config("./data/processed/label_mappings.json")
    dataset = config["dataset"]

    model = MODEL.get(dataset,CNN_Local)()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralised_dataset()

    test_loss, test_acc = test(model, 
                               test_dataloader,
                               device,
                               [int(label) for label in mappings["id_to_label"].keys()],
                               )

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})