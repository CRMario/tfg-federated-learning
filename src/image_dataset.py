import json
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import Dataset
from PIL import Image

"""Base class for the image datasets. Purpose is lazy image loading."""
class ImageDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        with open('./data/processed/label_mappings.json', "r") as f:
            label_mappings = json.load(f)

        self.id_label = label_mappings["id_to_label"]
        self.label_id = label_mappings["label_to_id"]

    def __len__(self):
        return len(self.images)
    
    def _get_label_id(self, label):
        if isinstance(label, str):
            return self.label_id[label]
        return label
    
    def _apply_transforms(self,image):
        return self.transforms(image)
    
    def get_label_name_map(self):
        return self.id_label

"""Image dataset for local images. Assumes 
images can be big and resizes them to 224x224"""
class ImageDatasetLocal(ImageDataset):

    transforms = Compose(
                    [Resize((224,224)),
                    ToTensor(), 
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, id):
        # only load the image when necessary (when we get the image)
        image = Image.open(self.images[id]).convert("RGB")
        image = self._apply_transforms(image)

        label = self._get_label_id(self.labels[id])

        return {"img": image, "label": torch.tensor(label, dtype=torch.long)}
    
    
"""Image dataset for BloodMNIST dataset."""
class ImageDatasetBloodMNIST(ImageDataset):

    transforms = Compose(
                        [ToTensor(), 
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, id):
        # only load the image when necessary (when we get the image)
        image = Image.fromarray(self.images[id])
        image = self._apply_transforms(image)

        label = self._get_label_id(self.labels[id])

        return {"img": image, "label": torch.tensor(label, dtype=torch.long)}
    
"""Image dataset for MNIST dataset."""
class ImageDatasetMNIST(ImageDataset):

    transforms = Compose([
        ToTensor(),
    ])

    def __getitem__(self, id):
        image = Image.fromarray(self.images[id]) # MNIST uses numpy arrays
        image = self._apply_transforms(image)

        label = self._get_label_id(self.labels[id])

        return {"img": image, "label": torch.tensor(label, dtype=torch.long)}
    
IMAGE_DATASET = {
    "local": ImageDatasetLocal,
    "bloodmnist": ImageDatasetBloodMNIST,
    "mnist": ImageDatasetMNIST,
}