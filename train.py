import logging
import math
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

LOGGER = logging.getLogger('train')

# Set paths and parameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
N_EPOCHS = 100

CLASSES = ['house', 'apartment_building-outdoor']  # We could add a third class: 'street' (other).
N_TRAIN_SAMPLES = 10000
N_VALIDATION_SAMPLES = 200

USE_CACHE = True

TRAIN_DIR = "input_files/train_building"  # Includes data on each class grouped in a subdirectory.
VALIDATION_DIR = "input_files/val_building"  # Link: http://places2.csail.mit.edu/download.html

OUTPUT_DIR = f"output_files/{str(datetime.utcnow())[:16]}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_BOTTLENECK_FEATURES_PATH = f"input_files/train_bottleneck_features_{CLASSES}.npy"
VAL_BOTTLENECK_FEATURES_PATH = f"input_files/val_bottleneck_features_{CLASSES}.npy"

TOP_MODEL_WEIGHTS_PATH = OUTPUT_DIR + "bottleneck_weights.h5"
FINAL_MODEL_PATH = OUTPUT_DIR + "final_building_model.h5"

MODEL_CHECKPOINT_DIR = OUTPUT_DIR + "model_checkpoints/"
MODEL_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR + "epoch_{epoch:02d}_val_acc_{val_acc:.2f}.h5"
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)


def compute_bottleneck_features():
    """
    Cache the feature values of the data in the bottleneck layer in order to speed up training.

    Returns:
        The values of both the train and validation set features in the bottleneck layer.
    """
    # Build the VGG16 network.
    logging.info("Loading VGG16 model with ImageNet weights.")
    model = VGG16(include_top=False, weights='imagenet')

    LOGGER.info("Setting up train data generator.")
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode=None,
        shuffle=False)

    LOGGER.info("Class indices train set: %s", train_generator.class_indices)
    LOGGER.info("Getting bottleneck features of the train set.")
    t0 = time.perf_counter()
    train_bottleneck_features = model.predict_generator(train_generator,
                                                        N_TRAIN_SAMPLES // BATCH_SIZE)
    LOGGER.info("Train bottleneck features computed in %d seconds.", time.perf_counter() - t0)

    LOGGER.info("Setting up validation data generator.")
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode=None,
        shuffle=False)

    LOGGER.info("Class indices validation set: %s", val_generator.class_indices)
    LOGGER.info("Getting bottleneck features validation set.")
    t0 = time.perf_counter()
    val_bottleneck_features = model.predict_generator(val_generator,
                                                      N_VALIDATION_SAMPLES // BATCH_SIZE)
    LOGGER.info("Validation bottleneck features computed in %d seconds.", time.perf_counter() - t0)

    return train_bottleneck_features, val_bottleneck_features


def get_bottleneck_features(use_cache=False):
    """
    Get bottleneck features from either a cache, or by computing them from scratch.

    Args:
        use_cache: Boolean that indicates whether or not to use a cache.

    Returns:
        The values of both the train and validation set features in the bottleneck layer.
    """
    if use_cache:
        LOGGER.info("Loading cached bottleneck features.")
        train_bottleneck_features = np.load(TRAIN_BOTTLENECK_FEATURES_PATH)
        val_bottleneck_features = np.load(VAL_BOTTLENECK_FEATURES_PATH)

    else:
        LOGGER.info("No cache used. Computing bottleneck layers from scratch.")
        train_bottleneck_features, val_bottleneck_features = compute_bottleneck_features()
        LOGGER.info("Saving training bottleneck features in %s", TRAIN_BOTTLENECK_FEATURES_PATH)
        np.save(TRAIN_BOTTLENECK_FEATURES_PATH, train_bottleneck_features)
        LOGGER.info("Saving validation bottleneck features in %s", VAL_BOTTLENECK_FEATURES_PATH)
        np.save(VAL_BOTTLENECK_FEATURES_PATH, val_bottleneck_features)

    return train_bottleneck_features, val_bottleneck_features


def train_top_model(x_train, y_train, x_val, y_val):
    """
    Train the top layers of the model using the bottleneck features of the VGG16 model.
    Args:
        x_train: Training set features.
        y_train: Training set labels.
        x_val: Validation set features.
        y_val: Validation set labels.

    Returns:
        Trained top model.
    """
    # Define model architecture.
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=x_train.shape[1:]))
    model.add(layers.Dense(100, activation='selu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model using early stopping.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1,
                                   restore_best_weights=True, patience=20)
    callbacks_list = [early_stopping]

    LOGGER.info("Fitting top model.")
    model.fit(x_train, y_train,
              epochs=N_EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(x_val, y_val),
              callbacks=callbacks_list)

    LOGGER.info("Saving top model weights.")

    return model


def build_full_model():
    """
    Build the full model using trained top model layers.

    Returns:
        Full model with a VGG16 convolutional base and a trained top model.
    """
    # Build the VGG16 network.
    model = models.Sequential()
    model.add(VGG16(weights='imagenet', include_top=False,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    # Build the model to put on top of the VGG16 model.
    top_model = models.Sequential()
    top_model.add(layers.Flatten(input_shape=model.output_shape[1:]))
    top_model.add(layers.Dense(100, activation='selu'))
    top_model.add(layers.Dropout(0.5))  # TODO: Try training a model without or with a higher rate.
    top_model.add(layers.Dense(1, activation='sigmoid'))

    # Note that it is necessary to start with a fully-trained classifier, including the top
    # classifier, in order to successfully do fine-tuning.
    top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

    # Add the model on top of the convolutional base.
    model.add(top_model)

    # Show a summary of the model
    LOGGER.info(model.summary())

    return model


def get_data_generators():
    """
    Create data generators for the training set and validation set.

    Returns:
        Training and validation set data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator


def train_full_model(model, train_generator, validation_generator):
    """
    Train the full model and save the model object.

    Args:
        model: Full model that will be trained.
        train_generator: Data generator for the training set.
        validation_generator: Data generator for the validation set.

    Returns:
        History of the training process.
    """
    # Define callbacks and compile the model.
    early_stopping = EarlyStopping(monitor='val_loss',
                                   verbose=1,
                                   restore_best_weights=True,
                                   patience=20)
    model_checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_PATH,
                                       monitor='val_loss',
                                       mode='min',
                                       verbose=1)
    callbacks_list = [early_stopping, model_checkpoint]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # Fine-tune the model.
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.samples / train_generator.batch_size),
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.samples / validation_generator.batch_size),
        callbacks=callbacks_list,
        verbose=1)

    # Save the model
    model.save(FINAL_MODEL_PATH)

    return history


def plot_train_process(history):
    """
    Plot the training process.

    Args:
        history: History of the training process.
    """
    # Plot training results
    accuracy = history.history['accuracy']
    LOGGER.info("Model accuracy: %f", accuracy)
    val_accuracy = history.history['val_accuracy']
    LOGGER.info("Model validation accuracy: %f", val_accuracy)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(N_EPOCHS, accuracy, 'b', label='Training accuracy')
    plt.plot(N_EPOCHS, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(N_EPOCHS, loss, 'b', label='Training loss')
    plt.plot(N_EPOCHS, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def main():
    # Cache the feature values of the bottleneck layer.
    train_bottleneck_features, val_bottleneck_features = get_bottleneck_features(
        use_cache=USE_CACHE)

    # Define the labels of the train and validation set, which are just zeroes for the first class
    # (house) and ones for the second class (apartment building).
    y_train = np.array([0] * int(N_TRAIN_SAMPLES / 2) + [1] * int(N_TRAIN_SAMPLES / 2))
    y_val = np.array([0] * int(N_VALIDATION_SAMPLES / 2) + [1] * int(N_VALIDATION_SAMPLES / 2))

    # Train top model using bottleneck features as input and store weights.
    top_model = train_top_model(train_bottleneck_features, y_train, val_bottleneck_features, y_val)
    top_model.save_weights(TOP_MODEL_WEIGHTS_PATH)

    # Build the full model.
    model = build_full_model()

    # Freeze the first 25 layers (up until the last convolutional block).
    for layer in model.layers[:25]:
        layer.trainable = False

    # Get data generators, train the full model and visualize results.
    train_generator, validation_generator = get_data_generators()
    history = train_full_model(model, train_generator, validation_generator)
    plot_train_process(history)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
