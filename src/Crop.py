import numpy as np
from matplotlib import pyplot as plt
from NumpyConv import image_to_ndarray, plot_im
import tensorflow as tf
from multiprocessing import Pool 

def main():
    plot_rect("data/label_0/58849.108186430596_DM_51.27_beam_2.jpg")

def discover_unaccounted(): #goes through the dataset and lists the file paths of graphs whose dimensions have't been documented
    rfi_names = tf.io.gfile.glob("data/label_0/*.jpg") #assumes testing from 'root' directiory
    frb_names = tf.io.gfile.glob("data/label_1/*.jpg")
    files = rfi_names + frb_names

    #use parallel processing to open files and determine their dimensions
    dims = None  
    with Pool() as pool:
        results = pool.map(get_dims, files)
        dims = list(results)

    if dims == None:
        print("error")
        return -1

    #tally the number of unaccounted dimensions in the dataset
    unaccounted = {}
    for dim in dims:
        if dim[1] == (689, 944): #dimensions of know plots
            continue
        if dim[1] == (631, 922):
            continue
        if dim[1] in unaccounted:
            unaccounted[dim[1]].append(dim[0])
        else:
            unaccounted[dim[1]]=[dim[0]]
    
    with open("data/dimensions.txt", "w") as f:
        print(unaccounted, file=f)
    
    print("done")


def plot_rect(file_name): #used for visualizing cropping block of picture
    img = image_to_ndarray(file_name)
    
    plot_im(img)
    print(img.shape)

    crop_1 = (369, 86, 268, 359)  # for graphs with shape (689, 944, 3),
    crop_2 = (337, 83, 245, 351) # for graphs with shape (631, 922, 3)

    image = np.zeros_like(img)
    if img.shape[0] == 689:
        image = draw_rect(img, crop_1)
    elif img.shape[0] == 631:
        image = draw_rect(img, crop_2)
    else:
        print("New shape")

    plot_im(image)

def draw_rect(img, co_ords): #co_ords (top_left_pixel_height, width, target height, target width)
    img = np.array(img)
    img[ co_ords[0], co_ords[1]:(co_ords[1]+co_ords[3]), : ] = 0
    img[ co_ords[0]+co_ords[2], co_ords[1]:(co_ords[1]+co_ords[3]), : ] = 0
    img[ co_ords[0]:(co_ords[0]+co_ords[2]), co_ords[1], :] = 0
    img[ co_ords[0]:(co_ords[0]+co_ords[2]), co_ords[1]+co_ords[3], :] = 0
    return img

def get_dims(file):
    img = image_to_ndarray(file)
    h, w, _ = img.shape
    return (file, (h,w))

if __name__ == "__main__":
    main()
