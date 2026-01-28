from pathlib import Path

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
    data_path : str
        String with the path to the directory that contains the image directories.

    Returns
    -------
    result: dict[str,List[str]]
        A dictionary mapping each class label to a list with the paths of the
        images that belong to the class.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {path}")
    
    result = {}
    for image_class in path.iterdir():

        # Create a dictionary of each of the labels
        result[image_class.name] = []
        
        if not image_class.is_dir() or image_class.name.startswith('.'):
            continue

        image_paths = []

        for image_path in image_class.iterdir():

            if image_path.suffix.lower() in EXTENSIONS:
                image_paths.append(str(image_path))

        if (image_paths):
            result[image_class.name] = image_paths

    return result