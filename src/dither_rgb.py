import numpy as np
from PIL import Image
import cv2

# GREYSCALE = False
# img_name = r"C:\Users\dasha\Desktop\bakalarka_data\final_data\2D\3422.jpeg"
# img_np1 = cv2.imread(img_name)
# print(img_np1.shape)
# img = Image.open(img_name)
# img_np = np.array(img)
# print(img_np.shape)
# img_name = r"C:\Users\dasha\Desktop\bakalarka_data\cropped_datasets\train_val_data\train\dec_masks\2.jpeg"

# Read in the image, convert to greyscale.
# img = Image.open(img_name)
# if GREYSCALE:
#     img = img.convert('L')

# new_width, new_height = img.size

def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.

    """

    return np.round(old_val * (nc - 1)) / (nc - 1)



def fs_dither(img_name, nc):
    """
    Floyd-Steinberg dither the image img into a palette with nc colours per
    channel.

    """
    img = Image.open(img_name)
    new_width, new_height = img.size

    arr = np.array(img, dtype=float) / 255

    for ir in range(new_height):
        for ic in range(new_width):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            # In this simple example, we will just ignore the border pixels.
            if ic < new_width - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < new_height - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < new_width - 1:
                    arr[ir+1, ic+1] += err / 16

    carr = np.array(arr/np.max(arr, axis=(0,1)) * 255, dtype=np.uint8)
    control_set = []
    for i in range(new_height):
        for j in range(new_width):
            control_set.append(carr[i,j])

    control0 = set([el[0] for el in control_set])
    control1 = set([el[1] for el in control_set])
    control2 = set([el[2] for el in control_set])

    print(len(control0), len(control1), len(control2), )
    print(len(np.unique(control_set, axis=0)))
    return Image.fromarray(carr)


# def palette_reduce(img, nc):
#     """Simple palette reduction without dithering."""
#     arr = np.array(img, dtype=float) / 255
#     arr = get_new_val(arr, nc)

#     carr = np.array(arr/np.max(arr) * 255, dtype=np.uint8)
#     return Image.fromarray(carr)

# for nc in (2, 3, 4, 8, 16):
#     print('nc =', nc)
#     dim = fs_dither(img, nc)
#     dim.save('dimg-{}.jpg'.format(nc))
#     # rim = palette_reduce(img, nc)
#     # rim.save('rimg-{}.jpg'.format(nc))
# nc = 8
# print('nc =', nc)
# dim = fs_dither(img_name, 8)
# dim.save('try.jpg'.format(nc))



# dith_img = np.asarray(cv2.imread(r"C:\Users\dasha\Desktop\py_projects\bakalarka\src\gray_256c_2.jpg"))
# unique_pixels = dith_img.reshape(-1, dith_img.shape[2])

# dither_img2 = np.asarray(cv2.imread(r"C:\Users\dasha\Desktop\py_projects\bakalarka\src\try.jpg"))
# unique_pixels2 = dither_img2.reshape(-1, dither_img2.shape[2])

# arr = (unique_pixels == unique_pixels2)
# print(np.min(arr), np.max(arr))
# # print(np.unique(dith_img, axis=0)[0])
# # print(len(np.unique(dith_img, axis=1)))
# # print(unique_pixels)
# print(len(unique_pixels))