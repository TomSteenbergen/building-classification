"""
- Fetch file with address list
- Download data from Google Street View API using address list
- Load model and predict
"""
import csv
from pathlib import Path
import sys
import logging
import requests
from requests.utils import requote_uri
from keras.models import load_model

LOGGER = logging.getLogger('train')

# Set paths and API specifications
MODEL_PATH = "output_files/2019-09-08 20:32/final_building_model.h5"
IMAGE_DIR = "input_files/street_view_images"

BASE_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview?"
BASE_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata?"
SIZE = 256


def _format_street_view_url(base_url, location, api_key):
    url = base_url + f"size={SIZE}x{SIZE}" + f"&loc={location}" + f"&key={api_key}"
    url = requote_uri(url)

    return url


def check_image_status(location, api_key):
    url = _format_street_view_url(BASE_METADATA_URL, location, api_key)
    response = requests.get(url)

    if response.content["status"] == "OK":
        return True

    else:
        LOGGER.info("Image for location %s gives status code %s", location,
                    response.content["status"])
        return False


def get_and_save_street_view_image(location, api_key):
    url = _format_street_view_url(BASE_IMAGE_URL, location, api_key)
    local_path = Path(IMAGE_DIR) / (location + ".jpg")

    response = requests.get(url)

    with open(local_path, "wb+") as file:
        file.write(response.content)


def main():
    # Load model, addresses, and api key.
    model = load_model(MODEL_PATH)
    key = ...
    addresses = ...

    # Loop over all addresses, check if their image exists, and if so, download it.
    for address in addresses:
        if check_image_status(address, key):
            get_and_save_street_view_image(address, key)

    # Predict for each image that we have fetched and store the results.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
