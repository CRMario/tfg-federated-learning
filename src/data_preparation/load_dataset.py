from pathlib import Path
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_images(data_path):
    """
    Loads images and labels from the dataset in data_path.

    The directory is expected to have the following format:
        data_path/
            class_1/
                img1.jpg
                img2.png
                ...
            class_2/
                img3.jpeg
                img4.jpg
                ...
    
    Parameters
    ----------
    data_path : String
        String with the path to the directory that contains the image directories.

    Returns
    -------
    images : array[Image]
        An array containing the loaded images from the directories.    
    class : array[String]    
        An array containing the class of the image. 
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {path}")
    
    images = []
    classes = []
    for image_class in Path(data_path).iterdir():
        
        if not image_class.is_dir():
            continue

        for image_path in image_class.iterdir():

            if not image_path.suffix.lower not in EXTENSIONS:
                continue

            img = Image.open(image_path).convert("RGB")
            images.append(img)
            classes.append(image_class.name)

    return images, classes