import requests
import logging
from datetime import datetime
import os
import json
import unidecode

logging.basicConfig(
    level=logging.INFO,
    filename='met_data_extraction.log',
    filemode='w',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)

logger = logging.getLogger('met_extraction')

def create_dir(dir_name:str)->None:
    '''Creates a new folder if it does not already exist.'''
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return

def format_string(s:str):
    '''Gets rid of diacriticts and punctution.'''

    format_str = unidecode.unidecode(s)
    bad_chars = [' ', '.', ',', ';', '\'', '-', ':', '\r', '\n', '\t', 
                 '\b', '\a', '?', '!', '(', ')', '\"', '/', '<', '>', '*']
    for c in bad_chars:
        format_str = format_str.replace(c, '_')
    return format_str

def save_img(url:str, img_name:str):
    '''Saves image from url to the img_name.'''
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"}

    response = requests.get(url, headers=headers)
    if response.ok:
        with open(img_name, "wb") as f:
            f.write(response.content)
    return response.ok

def get_all_objects():
    '''Gets objects from the metmus, returns object IDs.'''
    base_url_all_objects = 'https://collectionapi.metmuseum.org/public/collection/v1/objects'
    response_all_objects = requests.get(f'{base_url_all_objects}')
    response_all_objects_json = response_all_objects.json()
    total_artworks = response_all_objects_json['total']
    logger.info(f'[OBJECTS-IDS-EXTRACTION] - Qty of extracted Artworks: {total_artworks}')
    return response_all_objects_json['objectIDs']

def get_img(id:int):
    '''Downloads image with id and saves it to folder with the name of themedium of the artwork, for future sorting.'''
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
    base_dir = r"C:\Users\dasha\Desktop\bakalarka_data\metmus3"

    img_object_url = f'{base_url}{str(id)}'
    response = requests.get(img_object_url)
    if not response: return

    content = json.loads(response.text)

    if content['isPublicDomain'] and len(content['primaryImageSmall'])>0:
        classification = format_string(content["classification"])
        if len(classification) > 0:
            print(content['classification'])
            img_dir = os.path.join(base_dir, classification)
            create_dir(img_dir)
            img_path = os.path.join(img_dir, str(id)+'.jpeg')
            img_url = content['primaryImageSmall']
            resp = save_img(img_url, img_path)
        else:
            medium = format_string(content['medium'])
            if len(medium) > 150 or len(medium) == 0: return 
            print(content['medium'])
            img_dir = os.path.join(base_dir, medium)
            create_dir(img_dir)
            img_path = os.path.join(img_dir, str(id)+'.jpeg')
            img_url = content['primaryImageSmall']
            resp = save_img(img_url, img_path)

    return


def main():
    start_time = datetime.now()

    artworks_ids_all = get_all_objects()
    # 121_774 - 400_000, 453_271 - end???
    for i in range(453271, len(artworks_ids_all)):
        print(artworks_ids_all[i], i)
        get_img(artworks_ids_all[i])


    time_elapsed = datetime.now() - start_time
    logger.info(f'[ARTWORK-PAYLOAD-EXTRACTION] - Time elapsed (hh:mm:ss.ms) {time_elapsed}')

if __name__ == '__main__':
    main()