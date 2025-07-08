import numpy as np
import evaluation as my_eval

THRESHOLD = 0.05
class_names = ['2D', 'Arch_plans', 'Architecture', 'Exhibition', 'NOT_IMG', 'WITHOUT_LABEL_PHOTO',
                'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general',
                'dec_jewelry', 'dec_masks', 'dec_medal_plaquettes', 'dec_utensils', 'sculpture'] # for pytorch model

dif_arr = []
# class_names = ['2D', 'Arch_plans','Architecture','Exhibition','NOT_IMG','Sculpture','WITHOUT_LABEL_PHOTO',
#             'dec_books', 'dec_coins', 'dec_fabric', 'dec_fans', 'dec_furniture', 'dec_general',
#             'dec_jewelry', 'dec_masks', 'dec_medals_plaquettes', 'dec_utensils'] # for keras model

def sort_distribution(dist_pred:np.ndarray)->dict:
    '''Sorts classes according to their probability being true class, returns sorted dictionary of classes and corresponding probabilities.'''
    prob_class = {c:prob for c, prob in enumerate(dist_pred)}
    sorted_prob_class = sorted(prob_class.items(), key=lambda x:x[1], reverse=True)
    return sorted_prob_class

def is_uncertain_prob(sorted_dist:dict)->bool:
    '''Check if the difference between the probabilities of the two most 
    probable classes is less than a threshold, taht is an uncertain prediction.'''
    most_prob = sorted_dist[0][1]
    second_most_prob = sorted_dist[1][1]
    dif_arr.append(most_prob - second_most_prob)
    if (most_prob - second_most_prob) > THRESHOLD: return False
    return True

def contains_target(sorted_dist, t)->bool:
    '''Checks if the true class is in the 3 most probable classes predicted by the model.'''
    most_prob = sorted_dist[0][0]
    second_most_prob = sorted_dist[1][0]
    third_most_prob = sorted_dist[2][0]
    if int(t) == most_prob or int(t) == second_most_prob or int(t) == third_most_prob: return True
    return False

def analyze_predicted_distributions(pred_dists:np.ndarray, targets:np.ndarray):
    '''Analyzes prediction distribution and prints the statistics.'''
    certain_preds = []
    certain_targets = []
    uncertain_preds = []
    uncertain_targets = []
    count_contains_target_uncertain = 0
    count_contains_target_all = 0
    pure_acc = 0
    prediction = np.zeros(targets.shape)
    i = 0
    for pred, t in zip(pred_dists, targets):
        prediction[i] = np.argmax(pred)
        i+=1
        # if (np.argmax(pred) == int(t)): pure_acc +=1
        # pred_dict = sort_distribution(pred)
        # if contains_target(pred_dict, t): count_contains_target_all += 1

        # if is_uncertain_prob(pred_dict):
        #     if contains_target(pred_dict, t): count_contains_target_uncertain += 1
        #     uncertain_preds.append(pred)
        #     uncertain_targets.append(t)
        # else:
        #     certain_preds.append(np.argmax(pred))
        #     certain_targets.append(t)
    
    # print()
    # print('ENTIRE_SET_ACC:', pure_acc/len(targets))
    # print('contains target in top3:', count_contains_target_all/len(targets))
    # print("amount of uncertain predictions:", len(uncertain_preds))
    # print("percentage of uncertain predictions to the test set size:", len(uncertain_preds)/len(pred_dists))
    # print()
    # print("amount of uncertain predictions containing target in top 3:", count_contains_target_uncertain)
    # if (len(uncertain_preds)):
    #     print("percentage of containing target to uncertain predictions:", count_contains_target_uncertain/len(uncertain_preds))
    # print()
    # certain_preds = np.array(certain_preds)
    # certain_targets = np.array(certain_targets)
    # if certain_targets.shape[0] == 0: print("accuracy without uncertain class:", 0.0)
    # else: print("accuracy without uncertain class:", sum(certain_targets == certain_preds)/certain_targets.shape[0])
    # l_targets = my_eval.convert_nums_to_labels(certain_targets, class_names)
    # l_preds = my_eval.convert_nums_to_labels(certain_preds, class_names)
    # my_eval.display_confusion_matrix_with_one_dec(l_targets, l_preds)
    
    l_targets = my_eval.convert_nums_to_labels(targets, class_names)
    l_preds = my_eval.convert_nums_to_labels(prediction, class_names)
    my_eval.display_confusion_matrix_with_one_dec(l_targets, l_preds)
    return

def load_numpy_arr(arr_path:str)->np.ndarray:
    '''Loads numpy array from .npz file.'''
    arr = np.load(arr_path)['arr_0']
    return arr

def util(prediction_distributions_array_path:str, targets_array_path:str):
    '''Utility function, loads arrays to analyze, runs analysis function.'''
    dist_preds = load_numpy_arr(prediction_distributions_array_path)
    print('SIZE:', dist_preds.shape)
    # dist_preds /= 6 # for ensemble testing, to convert all values to the interval between 0 and 1
    print('MIN:', np.min(dist_preds), 'MAX:', np.max(dist_preds))
    print('threshold =', THRESHOLD)
    targets = load_numpy_arr(targets_array_path)
    analyze_predicted_distributions(dist_preds, targets)

    # darr = np.array(dif_arr)
    # print('AVEARGE:', np.mean(darr))
    # print('MIN:', np.min(darr))
    # print('MAX:', np.max(darr))

util(r"C:\Users\dasha\Desktop\py_projects\bakalarka\experiments_final\lbp4dim\predictions_dist.npz",
    r"C:\Users\dasha\Desktop\py_projects\bakalarka\experiments_final\lbp4dim\targets.npz")