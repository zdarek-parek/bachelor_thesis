import csv

def get_stat(file_path:str, label:str):
    '''Gets test statistics for each class (given as a folder).'''
    count_all = 0
    count1, count2, count3 = 0, 0, 0
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            count_all += 1
            class1 = row['class1']
            class2 = row['class2']
            class3 = row['class3']
            
            print(count_all)
            if class1 in label:
                count1 += 1
                continue
            if class2 in label:
                count2 += 1
                continue
            if class3 in label:
                count3 += 1
                continue
            

    
    print(f'All: {count_all}, calss1: {count1}, class2: {count2+count1}, class3:{count3+count2+count1}')
    print(f'All: {count_all}, calss1: {count1 / count_all}, class2: {(count2+count1)/count_all}, class3:{(count3+count2+count1)/count_all}')
    return

get_stat(r'bakalarka/src/ui_app/result/57_ other + architecture_result.csv', ['Architecture', 'WITHOUT_LABEL_PHOTO'])
# OVERALL: 3677
# CLASS1: 2314 - 62.9%
# CLASS2: 2916 - 79.3%
# CLASS3: 3256 - 88.5%