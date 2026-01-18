from data_preparation.load_dataset import load_images
from data_preparation.split_data import split_data_by_hospital
from datasites.generate_datasites import generate_datasites
from config.constants import *

def main():

    # Load the images of the dataset
    images = load_images(DATA_PATH)

    # Split the images amongst the hospitals
    hospitals_data = split_data_by_hospital(images,HOSPITALS)

    generate_datasites(HOSPITALS)

if __name__ == "__main__":
    main()