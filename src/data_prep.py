import cv2
from skimage.transform import pyramid_gaussian
import torchvision.transforms as transforms
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import shuffle
from sklearn.model_selection import train_test_split

import config_info as ci
import data_manipulation as dm


def resize_and_save_img(img_path:str, destination:str, num:int):
    '''Resizes image to 256x256 and saves new image in the destination.'''
    tr = transforms.Resize((256, 256))
    crop_img1 = tr.forward(Image.open(img_path))
    crop_img1.save(os.path.join(destination,  str(num)+".jpeg"))
    return

def resize_batch(source_dir:str, destination:str, num:int):
    '''Creates a new folder (destination) with cropped images from the source dir.'''
    imgs = os.listdir(source_dir)
    for i in range(len(imgs)):
        path = os.path.join(source_dir, imgs[i])
        resize_and_save_img(path, destination, num)
        num += 1
        print(num)
    return

# resize_batch(r"C:\Users\dasha\Desktop\bakalarka_data\data_periodicals\Photos", 
#              r"C:\Users\dasha\Desktop\bakalarka_data\final_dataset\WITHOUT_LABELS_PHOTO", 1368)

def create_data_dir(source_dir:str, destination:str, num:int):
    '''Takes a forlder with class folders and crops all the images in the class folders.'''
    dirs = os.listdir(source_dir)
    for i in range(len(dirs)):
        path = os.path.join(source_dir, dirs[i])
        num = resize_batch(path, destination, num)

    return num

def work_with_class_folder_npdataset(dir_path:str, images:list[np.ndarray], labels:list[str])->tuple:
    '''Appends given class images in dir_path to existing numpy arrays, images and labels.'''
    class_name = os.path.basename(dir_path)
    print(class_name)
    imgs = os.listdir(dir_path)

    for i in range(len(imgs)):
        # print(i)
        img_path = os.path.join(dir_path, imgs[i])
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_4_dim = dm.add_binirized_dim_to_img(img_path)
        images.append(img_4_dim)
        labels.append(ci.CLASS_NAMES_DICT[class_name])

    return images, labels

def create_np_dataset(classes_dir:str, dataset_path:str):
    '''Creates numpy dataset with images and labels out of the images in 
    the classes_dir divided into folders corresponding to the classes.'''
    images = []
    labels = []

    classes = os.listdir(classes_dir)
    for cl in classes:
        images, labels = work_with_class_folder_npdataset(os.path.join(classes_dir, cl), images, labels)

    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)

    np.savez_compressed(dataset_path, images = images, labels = labels)
    return

def load_np_dataset(dataset_path:str)->tuple:
    '''Loads numpy dataset from .npz file.'''
    dataset = np.load(dataset_path)
    # print(dataset['images'].shape)
    # print(dataset['labels'].shape)
    return dataset['images'], dataset['labels']

def shuffle_data(dataset_path:str)->tuple:
    '''Shuffles numpy dataset loaded from .npz file and returns shuffled numpy arrays, images and labels.'''
    images, labels = load_np_dataset(dataset_path)
    inds = list(range(labels.shape[0]))
    shuffle(inds)
    images_shuffled = images[inds, :, :, :]
    labels_shuffeled = labels[inds,]

    return images_shuffled, labels_shuffeled

def create_train_test_set(images:np.ndarray, labels:np.ndarray)->tuple:
    '''Splits given images and labels into train and test datasets.'''
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, stratify=labels)
    return train_data, train_labels, test_data, test_labels

def split_test_within_one_class(class_name:str, class_path:str, test_folder:str, test_split:float):
    '''Splits images of one class into two sets, every 1/test_split image is
    saved in the new folder (test_class_path) and the rest of the images 
    stays in the original class folder.'''
    test_class_path = os.path.join(test_folder, class_name)
    if not os.path.exists(test_class_path):
        os.mkdir(test_class_path)
    
    imgs = os.listdir(class_path)
    # test_imgs_number = int(len(imgs)*test_split)

    for i in range(0, len(imgs), int(1/test_split)):
        old_name = os.path.join(class_path, imgs[i])
        new_name = os.path.join(test_class_path, imgs[i])
        os.rename(old_name, new_name)

    return

def create_test_folder(test_split:float, folder:str, test_folder:str):
    '''Splits a dataset in the given folder into two datasets according to the test_split. 
    Dataset in the folder becomes of size (1-test_split)\*original_dataset_size, 
    the result dataset of size test_split\*original_dataset_size is in the test_folder.'''
    class_folders = os.listdir(folder)
    for cf in class_folders:
        class_path = os.path.join(folder, cf)
        split_test_within_one_class(cf, class_path, test_folder, test_split)
    return

# create_test_folder(0.5, r"C:\Users\dasha\Desktop\bakalarka_data\final_dataset\test_data\test", 
#                         r"C:\Users\dasha\Desktop\bakalarka_data\final_dataset\train_val_data\val")

def apply_dithering_to_imgs(imgs_path:str, num:int, dest:str):
    '''Applies dithering to the input images in the imgs_path and saves new images to the dest.'''
    tr = transforms.Resize((256, 256))
    imgs = os.listdir(imgs_path)
    for i in range(len(imgs)):
        dither_img = dm.dither_img(os.path.join(imgs_path, imgs[i]))
        # crop_img1 = tr.forward(dither_img)
        # crop_img1.save(os.path.join(dest, str(num)+".jpeg"))
        # # dither_img.save(os.path.join(dest, str(num)+".jpeg"))
        num += 1
        print(imgs[i])
    return num

# apply_dithering_to_imgs(r"C:\Users\dasha\Desktop\bakalarka_data\foreign_data\metmus_sorted\ARCHITECTURE_", 
#                         0, r"C:\Users\dasha\Desktop\bakalarka_data\metmus_dither\ARCHITECTURE_")

def apply_dithering_to_folders(folder_path:str, num:int, dest:str):
    '''Applies dithering to the images in the folders that are in folder_path.'''
    dirs = os.listdir(folder_path)
    for i in range(len(dirs)):
        num = 0
        dest_dir = os.path.join(dest, dirs[i])
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        apply_dithering_to_imgs(os.path.join(folder_path, dirs[i]), num, dest_dir)
    return

apply_dithering_to_folders(r"C:\Users\dasha\Desktop\bakalarka_data\final_data", 
                           0, r"")

def create_4dim_class(source:str, dest:str):
    '''Creates 4-dim images out of images in the source and saves them in the dest.'''
    imgs = os.listdir(source)

    for im in imgs:
        img_path = os.path.join(source, im)
        name = im.split('.')[0] + '.npy'
        print(name)
        new_img_path = os.path.join(dest, name)
        new_img = dm.edge_detection_canny(img_path)
        np.save(new_img_path, new_img)

    return 

def create_4dim_dataset(dataset_path:str, dest:str):
    '''Creates a 4-dim dataset out of the dataset folder, dataset_path, 
    that contains class folders. The result dataset is in dest in a form of class folders.'''
    class_dirs = os.listdir(dataset_path)

    for i in range(len(class_dirs)):
        print(class_dirs[i])
        class_dest = os.path.join(dest, class_dirs[i])
        os.mkdir(class_dest)
        class_source = os.path.join(dataset_path, class_dirs[i])
        create_4dim_class(class_source, class_dest)

    return

def rename(folder_path:str, dest:str):
    num = 0
    dirs = os.listdir(folder_path)
    for d in dirs:
        d_path = os.path.join(folder_path, d)
        imgs = os.listdir(d_path)
        for img in imgs:
            print(num)
            old_path = os.path.join(d_path, img)
            new_path = os.path.join(dest, f'{str(num)}.jpeg')
            os.rename(old_path, new_path)
            num += 1
    return

# rename(r"C:\Users\dasha\Desktop\bakalarka_data\metmus_dither\UTENSILS_", r"C:\Users\dasha\Desktop\bakalarka_data\metmus_dither\datasets\dec_utensils")

