import random as rand

def split_data_by_hospital(images,hospitals,seed=42):
    """
    Splits the given images between the hospitals by shuffling the lists
    of images. Each hospital is given an equal amount of images of each label.

    Parameters
    ----------
    images : Dict[str,List[Image]]
        A dictionary where the key is the label and the value is a
        list containing all the images that were assigned to that label.

    hospitals : List[str]
        A list with the name of the hospitals.

    seed : int
        An integer that initializes the rand.
        
    Returns
    -------
    data_splits: Dict[str,List[Tuple[Image,str]]]
        A dictionary mapping each hospital to a list containing tuples
        of images alongside their given label.
    """

    rand.seed(seed)
    # Create a dictonary for each of the hospitals
    data_splits = {}
    for hospital in hospitals:
        data_splits[hospital] = []
    # Equally split the images amongst the hospitals
    n_hospitals = len(hospitals)
    for label, imgs in images.items():
         # Shuffle each list of images
        rand.shuffle(imgs)
        for i, img in enumerate(imgs):
            assigned_hospital = hospitals[i % n_hospitals]
            data_splits[assigned_hospital].append((img,label))

    return data_splits