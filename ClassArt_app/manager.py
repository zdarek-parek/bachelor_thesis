import csv
import requests
import time
import os


import class_tool as ct

last_request_time = 0.0

def check_time():
    '''Checks when the last request was. If it was less than 3 seconds ago, it sleeps the difference.'''
    time_constraint = 3
    current_time = time.time()
    if (current_time - last_request_time) < time_constraint:
        time.sleep(time_constraint - (current_time - last_request_time))

class Manager:
    '''The class is for input and output manipulation.'''
    def __init__(self):
        self.CSV_IMG_ID = 'item'
        self.CSV_IMG_ADDR = 'imageAddr'
        self.class_tool = ct.ClassificationTool()
        self.img_path_to_classify = "img_to_classify.jpeg"
        self.resized_img_path_to_classify = "resized_img_to_classify.jpeg"
        self.dummy_top_dist = [('', 0.0) for _ in range(3)]
        self.create_dir('result')
        # self.open_result_csv()

    
    def open_result_csv(self, file_name:str):
        '''Opens the CSV file that stores the results of the pipeline.'''
        self.csv_to_write = open(os.path.join('result', f'{file_name}_result.csv'), 'w', encoding='utf-8')
        self.fieldnames = ['item', 'imageAddr', 'class1', 'prob1', 'class2', 'prob2', 'class3', 'prob3']
        self.writer = csv.DictWriter(self.csv_to_write, fieldnames=self.fieldnames)
        self.writer.writeheader()
        return
    
    def create_csv_entry(self, item:str, img_addr:str, top_dist_classes:list)->dict:
        '''Creates an ebtry to the result CSV file.'''
        return {
            'item' : item,
            'imageAddr' : img_addr,
            'class1' : top_dist_classes[0][0],
            'prob1' : top_dist_classes[0][1],
            'class2' : top_dist_classes[1][0],
            'prob2' : top_dist_classes[1][1],
            'class3' : top_dist_classes[2][0],
            'prob3' : top_dist_classes[2][1],
        }

    def create_dir(self, dir_name:str):
        '''Creates a directory with a given name, if it does not already exist.'''
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return
    
    def delete_img(self):
        '''Deletes the images that were created in the process of processing the input.'''
        if os.path.exists(self.resized_img_path_to_classify):
            os.remove(self.resized_img_path_to_classify)
        if os.path.exists(self.img_path_to_classify):
            os.remove(self.img_path_to_classify)
    

class CSVManager(Manager):
    def __init__(self):
        super().__init__()

    def process_csv(self, csv_path:str):
        '''Processes a single input CSV file and writes the results to the output CSV file.'''
        try:
            self.open_result_csv(os.path.basename(csv_path).split('.')[0])
            with open(csv_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    img_id = row[self.CSV_IMG_ID]
                    img_address = row[self.CSV_IMG_ADDR]

                    if img_address.startswith('http'):
                        success = self.save_img(self.img_path_to_classify, img_address)
                        if success:
                            top_dist_classes = self.class_tool.get_classification_rank(self.img_path_to_classify)
                            csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                            self.writer.writerow(csv_entry)
                        else:
                            csv_entry = self.create_csv_entry(img_id, img_address, self.dummy_top_dist)
                            self.writer.writerow(csv_entry)
                    else:
                        top_dist_classes = self.class_tool.get_classification_rank(img_address)
                        csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                        self.writer.writerow(csv_entry)
                
            self.csv_to_write.close()
        except Exception as e:
            print("An error occurred...", e)
            self.csv_to_write.close()
        return


    def work_with_csv(self, csvs:list[str]):
        '''Processes a list of input CSV files.'''
        for csv in csvs:
            print(csv)
            self.process_csv(csv)
        self.delete_img()
        return
    
    def save_img_unsafe(self, url:str, img_name:str)->bool:
        '''Downloads an image from the given URL.'''
        # headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"}
        check_time()
        time.sleep(1)
        response = requests.get(url)
        global last_request_time
        last_request_time = time.time()
        if response.ok:
            with open(img_name, "wb") as f:
                f.write(response.content)
        return response.ok
        
    def save_img(self, img_name:str, url:str)->bool:
        '''Calls a function that downloads an image from the URL, in case of an error enforces timeout (60 s) and tries again.'''
        try:
            response_ok = self.save_img_unsafe(url, img_name)
            return response_ok
        except Exception as e:
            print("Error occurred. The error is ", e)
            print("Let me try again...")
            time.sleep(60)
            response_ok = self.save_img_unsafe(url, img_name)
            return response_ok

class DIRManager(Manager):
    def __init__(self):
        super().__init__()
    

    def process_dir(self, dir_name:str):
        '''Processes a single directory, that contains images (PNG, JPEG, JPG), and writes the results to the output CSV file. 
        Ignores everything except for images.'''
        try:
            self.open_result_csv(os.path.basename(dir_name))
            imgs = os.listdir(dir_name)
            for img in imgs:
                img_path = os.path.join(dir_name, img)
                if (img_path.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    img_id = img
                    img_address = img_path

                    top_dist_classes = self.class_tool.get_classification_rank(img_address)
                    csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                    self.writer.writerow(csv_entry)
                else:
                    print(f'Ignoring {img_path}. Not an image.')
                
            self.csv_to_write.close()
        except Exception as e:
            print("An error occurred...", e)
            self.csv_to_write.close()
        return


    def work_with_dir(self, dir_path:str):
        '''Expects a directory which contains directories which contain images. 
        Calls a function that processes a single directory for each directory. Ignores everything except for directories.'''
        dirs = os.listdir(dir_path)
        for d in dirs:
            d_path = os.path.join(dir_path, d)
            if os.path.isdir(d_path):
                print(d_path)
                self.process_dir(d_path)
            else:
                print(f'Ignoring {d_path}. Not a directory.')
        self.delete_img()
        return