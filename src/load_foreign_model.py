from torchvision import models
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import os
from torch.utils.data import DataLoader

import numpy as np
import evaluation as my_eval
import custom_data_trans as cdt

PATH = r"C:\Users\dasha\Desktop\py_projects\bakalarka\experiment_results\4dim_50epochs_datasetsizebyclasssize_weight\torch_model.pt"
data_dir = r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\test_data"

def data_test_prep(data_dir:str, custom_dataset_flag:bool):
    '''Creates dataset out of the data in the data_dir.
    Returns a dataloader, dataset size and class names.'''
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if custom_dataset_flag:
        image_datasets = {x: cdt.my_dataset(os.path.join(data_dir, x)) for x in ['test']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes

    return dataloaders, dataset_sizes, class_names


def change_model_arch(input_4dim_flag:bool):
    '''Loads pretrained pytorch AlexNet model and modifies its architecture, returns modified model.'''
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    if input_4dim_flag:
        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        model.features = new_featres
    print(model)
    return model

def util(weights_path:str, input_4dim_flag:bool=True, custom_dataset_flag:bool=True):
    '''Loads the model to test, creates test dataset, displays confusion matrix of the test results.'''
    model = change_model_arch(input_4dim_flag)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu') ))
    model.eval()

    dataloaders, dataset_sizes, class_names = data_test_prep(data_dir, custom_dataset_flag)
    
    predictions = np.empty((0))
    targets = np.empty((0))
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            targets = np.concatenate((targets,labels.numpy()))
            predictions = np.concatenate((predictions, preds.numpy()))

    print('ACCURACY:', sum(targets == predictions)/targets.shape[0])

    l_targets = my_eval.convert_nums_to_labels(targets, class_names)
    l_preds = my_eval.convert_nums_to_labels(predictions, class_names)
    my_eval.display_confusion_matrix_with_one_dec(l_targets, l_preds)

# util(PATH)

