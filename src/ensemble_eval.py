import torch
import os
from torch.utils.data import DataLoader
import numpy as np

import evaluation as my_eval
import custom_data_trans as cdt
import ensemble_models as ems


data_dir = r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\test_data"
image_datasets = {x: cdt.my_dataset(os.path.join(data_dir, x))
                for x in ['test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=1,
                                            shuffle=True, num_workers=1)
            for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

def load_model(model_arch_func, weight_path:str):
    '''Loads and returns a pretrained pytorch model with loaded weights.'''
    model = model_arch_func()
    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device('cpu') ))
    model.eval()
    return model

def get_ensemble_evaluation(path_to_model_weights:str):
    '''Loads ensemble models, computes prediction distribution, 
    prediction (most probable class) and notes true class 
    for each input of the dataset (global, created at the beginning of the file).'''
    model_funcs = [ems.model1, ems.model2, ems.model3, ems.model4, ems.model5, ems.model6, ems.model7, ems.model8, ems.model9, ems.model10,
                ems.model11, ems.model12, ems.model13, ems.model14, ems.model15, ems.model16, ems.model17, ems.model18, ems.model19, ems.model20,
                ems.model21, ems.model22, ems.model23, ems.model24, ems.model25, ems.model26, ems.model27, ems.model28, ems.model29, ems.model30]

    models = []
    predictions = np.zeros((dataset_sizes['test']))
    predictions_dist = np.zeros((dataset_sizes['test'], len(class_names)))
    targets = np.empty((0))

    for i in range(len(model_funcs)):
        path_to_model_w = os.path.join(path_to_model_weights, str(i+1)+r"\torch_model.pt")
        m = load_model(model_funcs[i], path_to_model_w)
        models.append(m)
    print("number of models:", len(models))
    i = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            print(i)
            for mo in models:
                mo_pred = torch.nn.functional.softmax(mo(inputs), dim = 1).numpy() 
                mo_predicted_class = np.argmax(mo_pred)
                predictions_dist[i][mo_predicted_class] += 1

            predicted_class = np.argmax(predictions_dist[i])
            predictions[i] = predicted_class
            i += 1
            targets = np.concatenate((targets,labels.numpy()))

    return predictions, targets, predictions_dist

def convert_nums_to_labels(arr:np.ndarray)->list[str]:
    '''Converts numbers to labels according to the index of the class in the class list.'''
    arr_labels = [class_names[int(x)] for x in arr]
    return arr_labels


predictions, targets, pred_dists = get_ensemble_evaluation(r".\bakalarka\ensemble2")
np.savez("predictions_dist_30ms_new_ens_use.npz", pred_dists)
np.savez("targets_for_dist_30ms_new_ens_use.npz", targets)

print('ACCURACY', sum(targets == predictions)/targets.shape[0])

l_targets = convert_nums_to_labels(targets)
l_preds = convert_nums_to_labels(predictions)
my_eval.display_confusion_matrix_with_one_dec(l_targets, l_preds)
