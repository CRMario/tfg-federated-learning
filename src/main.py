from data_preparation.load_dataset import load_images
from data_preparation.split_data import split_data_by_hospital
from datasites.generate_datasites import generate_datasites
from datasites.generate_dataset import upload_data
from config.constants import *

def main():

    # Load the images of the dataset
    images = load_images(DATA_PATH)

    # Split the images amongst the hospitals
    hospitals_data = split_data_by_hospital(images,HOSPITALS)

    # Generate a datasite per hospital
    datasites = generate_datasites(HOSPITALS)

    # Upload the generated data to the corresponding datasite
    upload_data(datasites,hospitals_data)

if __name__ == "__main__":
    main()