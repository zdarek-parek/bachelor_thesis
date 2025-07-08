from skimage import feature
import cv2
import numpy as np
import os

radius = 3
n_points = radius * 8

def preprocess_image(img_path:str)->np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lbp_img = feature.local_binary_pattern(img, n_points, radius, method='uniform')
    # cv2.imwrite('lbp_image.jpeg', lbp_img)
    # print(type(lbp_img))
    return resize_image(lbp_img)


def resize_image(img:np.ndarray)->np.ndarray:
    reshaped_img = img.reshape(1, -1)
    # print(reshaped_img.shape)
    return reshaped_img

def get_dataset(dataset_path:str):
    classes = os.listdir(dataset_path)

    dataset = []
    labels = []

    for cl in classes:
        class_path = os.path.join(dataset_path, cl)
        class_imgs = os.listdir(class_path)
        for img in class_imgs:
            img_path = os.path.join(class_path, img)
            img_feature = preprocess_image(img_path)
            dataset.append(img_feature)
            labels.append(cl)
    
    print('DATASET SIZE', len(dataset))
    print('LABELS SIZE', len(labels))
    dataset = np.squeeze(np.array(dataset), axis=1)
    print(dataset.shape)
    return dataset, labels


# preprocess_image(r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\train_val_data\train\2D\1.jpeg")