import sys
import numpy as np
import cv2
from torchvision import transforms
import torch
import os
from collections import Counter
from PIL import Image
from skimage import feature
from torchvision import models
import torch.nn as nn

radius = 3
n_points = radius * 8


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class ClassificationTool:
    '''
    The class is a wrapper around the pipeline.
    '''
    def __init__(self):
        self.model = None
        self.path_to_model_weights = resource_path("torch_model.pt")
        self.model = self._load_model(self.path_to_model_weights)
        self.classes_names = ['2D', 'Arch_plans', 'Architecture', 'Exhibition', 'NOT_IMG', 'WITHOUT_LABEL_PHOTO', 'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general', 'dec_jewelry', 'dec_masks', 'dec_medal_plaquettes', 'dec_utensils', 'sculpture']

    def _load_model(self, weight_path:str):
        '''Loads the model, changes its architecture and loads the weights obtained in the training.'''
        model = models.alexnet()   
        model.classifier[-1] = nn.Linear(4096, 17, bias=True)

        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

        model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

        model.features = new_featres
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device('cpu') ))
        model.eval()
        return model
    

    def _transform_img(self, img_path:str):
        '''Preprocesses the image before it is fed to the model.'''

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
        contrast_stretched_ycrcb = cv2.merge([y_channel_stretched, cr_channel, cb_channel])
        contrast_stretched_image = cv2.cvtColor(contrast_stretched_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for lbp
        lbp = feature.local_binary_pattern(gray_img, n_points, radius, method='uniform')
        lbp = lbp.astype(np.uint8)

        new_img = np.append(contrast_stretched_image, np.expand_dims(lbp, 2), axis=2)
        new_tensor_img = transforms.ToTensor()(new_img)

        return new_tensor_img

    def _predict(self, img_path:str)->list:
        '''Outputs a probability distribution over 17 classes for the image.'''
        img_tensor = self._transform_img(img_path)
        t = torch.zeros((1, 4, 256, 256), dtype=torch.float32)
        t[0] = img_tensor
        with torch.no_grad():
            mo_pred = torch.nn.functional.softmax(self.model(t), dim = 1).numpy() #mo(t).numpy()
        pred = mo_pred.tolist()
        return pred[0]
    
    def _get_top3(self, pred_dist:list)->dict:
        '''Returns top-3 most probable classes.'''
        pred_dict = dict(enumerate(pred_dist))
        c = Counter(pred_dict)
        top3 = c.most_common(3)
        return top3
    
    def _convert_num_to_class(self, num:int)->str:
        '''Converts the number (0-16) used by the model into the corresponding class name.'''
        return self.classes_names[num]
    
    def get_classification_rank(self, img_path:str)->list:
        '''Utility function. Takes a path of the image and returns top-3 most probable classes predicted by the model.'''
        tr = transforms.Resize((256, 256))
        crop_img1 = tr.forward(Image.open(img_path))
        resized_img_path = 'resized_img_to_classify.jpeg'
        crop_img1.save(resized_img_path)

        pred_dist = self._predict(resized_img_path)
        top3 = self._get_top3(pred_dist)

        top3_dist_classes = [(self._convert_num_to_class(num), p) for (num, p) in top3]
        return top3_dist_classes
          
