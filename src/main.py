from data_preparation.load_dataset import load_images
from config.constants import *

def main():

    # Load the images of the dataset
    images, classes = load_images(DATA_PATH)

if __name__ == "__main__":
    main()
