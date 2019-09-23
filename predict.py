import csv
import json
import logging
import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
import requests
from keras.models import load_model
from keras.preprocessing import image
from requests.utils import requote_uri

LOGGER = logging.getLogger(__name__)

# Set paths and API specifications
MODEL_PATH = "output_files/2019-09-08 20:32/final_building_model.h5"
ADDRESSES_PATH = "input_files/addresses.csv"
IMAGE_DIR = "input_files/street_view_images"
PREDICTIONS_PATH = "output_files/predictions.csv"

CLASS_MAPPING = {0: "house", 1: "apartment"}

BASE_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview?"
BASE_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata?"
IMAGE_SIZE = 256


def _format_street_view_url(base_url, location, api_key):
    url = base_url + f"size={IMAGE_SIZE}x{IMAGE_SIZE}" + f"&location={location}" + f"&key={api_key}"
    url = requote_uri(url)

    return url


def check_image_status(location, api_key):
    url = _format_street_view_url(BASE_METADATA_URL, location, api_key)
    response = requests.get(url)
    status = json.loads(response.content)["status"]

    if status == "OK":
        return True

    elif status == "OVER_QUERY_LIMIT":
        LOGGER.info("Query limit of Google Street View API reached.")
        sys.exit()

    elif status == "REQUEST_DENIED":
        raise KeyError("Request denied. API key not valid.")

    elif status == "INVALID_REQUEST":
        raise ValueError("Request invalid. URL used: %s", url)

    else:
        LOGGER.warning("Image for location '%s' gives status code '%s'", location,
                       status)
        return False


def get_and_save_street_view_image(location, api_key):
    url = _format_street_view_url(BASE_IMAGE_URL, location, api_key)
    local_path = Path(IMAGE_DIR) / (location + ".jpg")

    response = requests.get(url)

    if local_path.exists():
        LOGGER.warning("%s already exists. Overwriting this file!", local_path)

    with open(local_path, "wb+") as file:
        file.write(response.content)

    return local_path


def main():
    # Get model and API key.
    model = load_model(MODEL_PATH)
    api_key = os.getenv("API_KEY")

    # Get all predicted addresses. If there aren't any, created the file with the right columns.
    if Path(PREDICTIONS_PATH).exists():
        predicted_addresses = pd.read_csv(PREDICTIONS_PATH)["address"].tolist()
    else:
        empty_df = pd.DataFrame({"address": [], "prediction": []})
        empty_df.to_csv(PREDICTIONS_PATH, index=False)
        predicted_addresses = []

    line_count = 0
    # Loop over all addresses.
    with open(ADDRESSES_PATH, "r") as address_file, open(PREDICTIONS_PATH, "a") as prediction_file:
        # Initialize csv reader and writer.
        address_reader = csv.reader(address_file)
        prediction_writer = csv.writer(prediction_file)

        # Loop over all addresses and keep track of addresses checked.
        for row in address_reader:
            if line_count == 0:
                LOGGER.info("Column names are %s", ", ".join(row))
                line_count += 1
                continue

            # Check if address is already predicted for. If so, continue to the next address.
            address = row[5]
            if address in predicted_addresses:
                continue

            # If not, try to fetch the image from Google Street View API.
            try:
                if check_image_status(address, api_key):
                    image_path = get_and_save_street_view_image(address, api_key)

            except Exception as e:
                LOGGER.info("Addresses fetched from API: %d", line_count - 1)
                raise e

            # Predict the image that we fetched using the model loaded.
            if image_path:
                img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                x = image.img_to_array(img) / 255
                x = np.expand_dims(x, axis=0)

                prediction = model.predict_classes(x)
                prediction_class = CLASS_MAPPING.get(prediction[0][0], None)

                if prediction_class:
                    prediction_writer.writerow([address, prediction_class])
                else:
                    raise KeyError("Could not map class using prediction %d", prediction)

                predicted_addresses.append(address)
                line_count += 1

                if line_count - 1 == 1000:
                    LOGGER.info("1000 requests made, exiting process now.")
                    sys.exit()

                elif line_count - 1 % 100 == 0:
                    LOGGER.info("%s requests made.", line_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
