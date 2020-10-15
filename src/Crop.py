import numpy as np
from matplotlib import pyplot as plt
from NumpyConv import image_to_ndarray, plot_im
import tensorflow as tf
from multiprocessing import Pool 
from numpy.random import randint

def main():
    rfi_names = tf.io.gfile.glob("data/label_0/*.jpg") #assumes testing from 'root' directiory
    frb_names = tf.io.gfile.glob("data/label_1/*.jpg")
    files = rfi_names + frb_names

    for i in randint(0, len(files)-1, 100):
        plot_rect(files[i])

    #since there are very few files of dimensions  (689, 943, 3), go through a few of them based on discover_unaccounted findings
    plot_rect("data/label_0/58849.108186430596_DM_51.27_beam_2.jpg")
    plot_rect("data/label_0/58849.0671146872_DM_386.55_beam_3.jpg")
    plot_rect("data/label_0/58849.0814282329_DM_208.15_beam_5.jpg")
    plot_rect("data/label_0/58849.062169635305_DM_64.16_beam_1.jpg")
    plot_rect("data/label_0/58849.089714327994_DM_74.91_beam_1.jpg")


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

    crop_1 = (369, 86, 268, 359)  # for graphs with shape (689, 944, 3) or (689, 943, 3)
    crop_2 = (337, 83, 245, 351) # for graphs with shape (631, 922, 3)

    if img.shape[0] == 689:
        image = draw_rect(img, crop_1)
    else:
        image = draw_rect(img, crop_2)
    
    plot_im(image, title=f"{img.shape}")

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
