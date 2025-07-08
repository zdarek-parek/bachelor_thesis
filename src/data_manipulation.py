import cv2
import numpy as np
from PIL import Image
import dither_rgb as drgb

def binarize_img(image:np.array)->np.array:
    '''Binarizes a given image and returns it.'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    _, threshold_image = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # print(threshold_image.shape)
    # cv2.imwrite('bin_dim.png', threshold_image)
    return threshold_image

def add_binirized_dim_to_img(img_path:str):
    '''Adds a binarized image as a next dimension to an input n-dim image 
    and returns the (n+1)-dim image.'''
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bin_dim = binarize_img(img)

    new_img = np.append(img, np.expand_dims(bin_dim, 2), axis=2)
    # print(new_img.shape)
    # cv2.imwrite('new_img.png', new_img)
    return new_img


def floyd_steinberg2(image):
    Lx, Ly = image.shape
    
    for j in range(Ly):
        for i in range(Lx):
            rounded = round(image[i,j]) 
            err = image[i,j] - rounded
            image[i,j] = rounded
            if i<Lx-1: image[i+1,j] += (7/16)*err
            if j<Ly-1:
                image[i,j+1] += (5/16)*err
                if i>0: image[i-1,j+1] += (1/16)*err
                if i<Lx-1: image[i+1,j+1] += (3/16)*err    
    return image

def floyd_steinberg(image):
    # https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid39.ipynb -- source
    if len(image.shape) == 2: return floyd_steinberg2(image)
    x, y, c = image.shape
    
    for j in range(y):
        for i in range(x):
            for c in range(c):
                oldpixel = image[i,j,c]
                newpixel = round(oldpixel) 
                image[i,j,c] = newpixel
                quant_error = oldpixel - newpixel
                if i<x-1: image[i+1,j,c] += (7/16)*quant_error
                if j<y-1:
                    image[i,j+1,c] += (5/16)*quant_error
                    if i<x-1: image[i+1,j+1,c] += (3/16)*quant_error 
                    if i>0: image[i-1,j+1,c] += (1/16)*quant_error
                       
    return image

def floyd_steinberg_new(image):
    '''Applies Floyd-Steinberg dithering to the given image and returns modified image.'''
    if len(image.shape) == 2: return floyd_steinberg_new2(image)
    x, y, c = image.shape
    
    for j in range(y):
        for i in range(x):
            for c in range(c):
                oldpixel = image[i,j,c]
                newpixel = round(oldpixel) 
                image[i,j,c] = newpixel
                quant_error = oldpixel - newpixel
                if i<x-1: image[i+1,j,c] += (7/16)*quant_error
                if i > 0 and j < y-1: image[i-1,j+1,c] += (3/16)*quant_error
                if j < y-1: image[i,j+1,c] += (5/16)*quant_error
                if i < x-1 and j < y-1: image[i+1,j+1,c] += (1/16)*quant_error                   
    return image

def floyd_steinberg_new2(image):
    '''Applies Floyd-Steinberg dithering to the given 2-dim image and returns modified image.'''
    x, y = image.shape
    
    for j in range(y):
        for i in range(x):
            oldpixel = image[i,j]
            newpixel = round(oldpixel) 
            image[i,j] = newpixel
            quant_error = oldpixel - newpixel
            if i<x-1: image[i+1,j] += (7/16)*quant_error
            if i > 0 and j < y-1: image[i-1,j+1] += (3/16)*quant_error
            if j < y-1: image[i,j+1] += (5/16)*quant_error
            if i < x-1 and j < y-1: image[i+1,j+1] += (1/16)*quant_error                   
    return image



# def dither_img(img_path:str):
#     '''Preprocesses the image and applies dithering to it and returns modified image.'''
#     img = Image.open(img_path)
#     img = np.array(img).astype(np.float32) / 255.
#     dither_img = floyd_steinberg_new(img.copy()) * 255
#     dither_img = dither_img.astype(np.uint8)
#     im = Image.fromarray(dither_img)
#     return im

def dither_img(img_name:str):
    img = Image.open(img_name)
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        print('GRAYSCALE IMG')
        # dither = drgb.fs_dither(img_name, 8)
    else:
        print('NOT')
        # dither = drgb.fs_dither(img_name, 4)
    
    return img_np

def edge_detection_canny(img_path:str):
    '''Adds edges of the given n-dim image as (n+1) dimension and returns the (n+1)-dim image.'''
    gray_img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (3,3), 0)
    edges = cv2.Canny(blur, threshold1=100, threshold2=200)
    new_img = np.append(cv2.imread(img_path), np.expand_dims(edges, 2), axis=2)
    return new_img