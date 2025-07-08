import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

def get_class_sizes(dataset_path:str)->Dict[str, int]:
    '''Computes the size of each class based on the dataset in the dataset_path (folder containing class folders).'''
    classes = os.listdir(dataset_path)
    class_counts = {}

    for c in classes:
        class_path = os.path.join(dataset_path, c)
        class_imgs = os.listdir(class_path)
        number_of_instances = len(class_imgs)

        class_counts[c] = number_of_instances

    return class_counts

def convert_to_dist(class_sizes:Dict[str, int])->Dict[str, float]:
    '''Divides each class size by the size of the dataset, returns dictionary of class names and corresponding size fractions.'''
    # keys = list(class_sizes.keys())
    dataset_size = sum(list(class_sizes.values()))
    for key in class_sizes:
        class_sizes[key] /= dataset_size
    return class_sizes

def assign_weights(class_dist:Dict[str, float])->Dict[str, float]:
    '''Reverse assignment of the weights to the class name, 
    the biggest class gets the smallest weight, the smallest class gests the biggest weight.'''
    sorted_dist = dict(sorted(class_dist.items(), key = lambda item: item[1]))
    keys = list(sorted_dist.keys())
    keys.reverse()
    values = list(sorted_dist.values())
    wdist = {}

    for i in range(len(keys)):
        wdist[keys[i]] = values[i]

    return wdist

def inverse_to_size_weights(dataset_path:str)->Dict[str, float]:
    '''Utility function that combines steps of the inverse weights assignment.'''
    class_sizes = get_class_sizes(dataset_path)
    class_dist = convert_to_dist(class_sizes)
    dist = assign_weights(class_dist)
    return dist

def total_by_num_weights(dataset_path:str, num:int=17)->Dict[str, float]:
    '''Computes weight for each class based on its size, on the size of the dataset and assignes the weight to each class.
    Each weight is equal to dataset_size / (class_size\*num). num is equal to number of classes, but can be changed if needed. '''
    class_sizes = get_class_sizes(dataset_path)
    dataset_size = sum(list(class_sizes.values()))
    dist = {}

    keys = list(class_sizes.keys())
    for k in keys:
        dist[k] = (1/class_sizes[k]) * (dataset_size/num)

    print(sum(list(dist.values())))
    print(dist)
    return dist

def util(dataset_path:str)->Dict[str, float]:
    '''Utility function that returns the dictionary with class names and corresponding weights.'''
    dist = inverse_to_size_weights(dataset_path)
    # dist = total_by_num_weights(dataset_path)
    return dist

counts = get_class_sizes(r"C:\Users\dasha\Desktop\bakalarka_data\final_data")
print(convert_to_dist(counts))