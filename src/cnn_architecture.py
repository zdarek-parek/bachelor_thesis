import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

import os
import config_info as ci
import data_prep as dp

def build_CNN():
    '''Returns AlexNet model.'''
    model = Sequential([
        layers.Rescaling(1./255, input_shape = (256, 256, 3)),
        layers.Conv2D(filters=62, kernel_size=(11,11), strides=4, activation="relu"),
        layers.MaxPool2D(pool_size=(3,3), strides=2),
        layers.Conv2D(filters=31, kernel_size=(5,5), padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=(3,3), strides=2),
        layers.Conv2D(filters = 31, kernel_size=(3,3), padding="same", activation="relu"),
        layers.Conv2D(filters = 31, kernel_size=(3,3), activation="relu"),
        layers.Conv2D(filters = 31, kernel_size=(3,3), activation="relu"),
        layers.MaxPool2D(pool_size=(3,3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(ci.NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def load_dataset_from_file(path:str)->tf.data.Dataset:
    '''Loads tf.data.Dataset from the path.'''
    dataset = tf.data.Dataset.load(path, compression="GZIP")
    return dataset

def create_dataset(data_dir_train:str, data_dir_val:str):
    '''Returns two tensorflow datasets, train and validation, created from 
    two input folders containing class folders.'''
    train_ds = keras.utils.image_dataset_from_directory(
    data_dir_train,
    # validation_split=0.2,
    # subset="training",
    seed=0,
    # shuffle = False,
    image_size=(ci.IMG_HEIGHT, ci.IMG_WIDTH),
    batch_size=ci.BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_val,
    # validation_split=0.2,
    # subset="validation",
    seed=123,
    # shuffle = False,
    image_size=(ci.IMG_HEIGHT, ci.IMG_WIDTH),
    batch_size=ci.BATCH_SIZE)

    # train_ds.save(path=ci.TRAIN_DATASET_PATH, compression="GZIP")
    # val_ds.save(path=ci.VAL_DATASET_PATH, compression="GZIP")
    return train_ds, val_ds

def train_and_save_model(model, train_data:np.ndarray, train_labels:np.ndarray, val_data:np.ndarray, val_labels:np.ndarray):
    '''Trains the model on the given numpy train and validation datasets and saves it.'''
    history = model.fit(
            train_data,
            train_labels,
            validation_data = (val_data, val_labels),
            batch_size = ci.BATCH_SIZE,
            epochs= ci.EPOCHS
            )
    
    model.save(ci.MODEL_PATH)
    print("Saved the model...")
    return history

def train_and_save_model_tf_dataset(model, train_ds:tf.data.Dataset, val_ds:tf.data.Dataset):
    '''Trains the model on the given tensorflow train and validation datasets and saves it.'''
    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ci.EPOCHS
            )
    
    model.save(ci.MODEL_PATH)
    print("Saved the model...")
    print(type(model))
    return history

def visualize_train_history(history):
    '''Saves the image displaying train and validation accuracy and loss over the epochs.'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(ci.EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(r'keras_train_history.png')
    # plt.show()
    return


def load_model(model_path:str):
    '''Loads and returns keras model from .keras file.'''
    model = tf.keras.models.load_model(model_path)
    return model

def predict_batch(test_path:str, model_path:str):
    '''Predicts the labels of the batch in the test_path, that contains class folders.
    Computes and prints the accuracy of the prediction.
    Saves the prediction distribution for every test item and its true class.'''
    model = load_model(model_path)
    # print(model)
    predictions_dist = np.zeros((2400, 17)) # 2400 test set size, 17 - number of classes
    targets = np.zeros((2400,)) # 2400 - test set size
    index = 0
    acc = 0
    # classes = os.listdir(test_path)
    # classes = ['2D', 'Arch_plans', 'Architecture', 'Exhibition', 'NOT_IMG', 'WITHOUT_LABEL_PHOTO',
    #             'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general',
    #             'dec_jewelry', 'dec_masks', 'dec_medal_plaquettes', 'dec_utensils', 'sculpture'] # classes order used for the pytorch model
    classes = ['2D', 'Arch_plans','Architecture','Exhibition','NOT_IMG','Sculpture','WITHOUT_LABEL_PHOTO',
               'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general',
               'dec_jewelry', 'dec_masks', 'dec_medals_plaquettes', 'dec_utensils'] # classes oredr used for the keras model
    for i in range(len(classes)):
        class_path = os.path.join(test_path, classes[i])
        imgs = os.listdir(class_path)
        for im in imgs:
            img_path = os.path.join(class_path, im)
            img = tf.keras.utils.load_img(
                img_path, target_size=(ci.IMG_HEIGHT, ci.IMG_WIDTH)
                )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            score_arr = score.numpy()
            print(index)

            predictions_dist[index] = score_arr
            targets[index] = i
            index += 1
            
            if np.argmax(score_arr) == i: acc += 1
    print('ACCURACY:', acc/len(targets))
    np.savez("predictions_dist_keras3dim_new_dith.npz", predictions_dist) # for future analysis
    np.savez("targets_keras3dim_new_dith.npz", targets)
    return


def util():
    '''Training pipeline from dataset creation to the training and history visualization.'''
    model = build_CNN()
    print("Finished building CNN...")
    tr_dataset, val_dataset = create_dataset(r"C:\Users\dasha\Desktop\bakalarka_data\cropped_datasets\train_val_data\train", 
                                            r"C:\Users\dasha\Desktop\bakalarka_data\cropped_datasets\train_val_data\val")
    print("Created dataset...")
    history = train_and_save_model_tf_dataset(model, tr_dataset, val_dataset)
    print("Finished training... Visualizing now...")
    visualize_train_history(history)
