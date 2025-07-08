import numpy as np

CLASS_NAMES = ['2D', 'Arch_plans', 'Architecture', 'Exhibition', 'NOT_IMG', 'sculpture', 'WITHOUT_LABEL_PHOTO', 
               'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general', 
               'dec_jewelry', 'dec_masks', 'dec_medal_plaquettes', 'dec_utensils']

CLASS_NAMES_DICT = {'2D':0, 'Arch_plans':1, 'Architecture':2, 'Exhibition':3, 'NOT_IMG':4, 'Sculpture':5, 'WITHOUT_LABEL_PHOTO':6, 
               'dec_books':7, 'dec_coins':8, 'dec_fabric':9, 'dec_fans':10, 'dec_furniture':11, 'dec_general':12, 
               'dec_jewelry':13, 'dec_masks':14, 'dec_medals_plaquettes':15, 'dec_utensils':16}

NUM_TO_CLASS = {0:'2D', 1:'Arch_plans', 2:'Architecture', 3:'Exhibition', 4:'NOT_IMG', 5:'Sculpture', 6:'WITHOUT_LABEL_PHOTO', 
               7:'dec_books', 8:'dec_coins', 9:'dec_fabric', 10:'dec_fans', 11:'dec_furniture', 12:'dec_general', 
               13:'dec_jewelry', 14:'dec_masks', 15:'dec_medals_plaquettes', 16:'dec_utensils'}

NUM_CLASSES = len(CLASS_NAMES)#17
EPOCHS = 15

BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256

DATA_PATH = r"C:\Users\dasha\Desktop\bakalarka_data\cropped_data"
MODEL_PATH = r"C:\Users\dasha\Desktop\py_projects\bakalarka\models\model_1try_final.keras"
# MODEL_PATH_4DIM = r"C:\Users\dasha\Desktop\py_projects\bakalarka\models\model_2.keras"
TRAIN_DATASET_PATH = r"C:\Users\dasha\Desktop\py_projects\bakalarka\datasets\train_dataset"
VAL_DATASET_PATH = r"C:\Users\dasha\Desktop\py_projects\bakalarka\datasets\val_dataset"

NP_DATASET_PATH = r"C:\Users\dasha\Desktop\py_projects\bakalarka\datasets\np_dataset.npz"
NP_DATASET_PATH_4DIM = r"C:\Users\dasha\Desktop\py_projects\bakalarka\datasets\np_dataset_4dim.npz"

def convert_index_to_class(index:int)->str:
    return NUM_TO_CLASS[index]

def find_top_most_probable_classes(score:np.ndarray, top:int=3)->list[str]:
    inds = np.argpartition(score, -top)[-top:]

    ind_score = {}
    for ind in inds:
        ind_score[ind] = score[ind]

    sorted_inds = dict(sorted(ind_score.items(), key = lambda ind_s:ind_s[1], reverse=True))
    res_classes = []
    for ind in sorted_inds.keys():
        res_classes.append(CLASS_NAMES[ind])
    return res_classes