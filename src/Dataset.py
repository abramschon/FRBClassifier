import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def main():
    #get file paths for the rfis and frbs
    rfi_names = tf.io.gfile.glob("PlotsForMLFRBs/label_0/*.jpg")
    frb_names = tf.io.gfile.glob("PlotsForMLFRBs/label_1/*.jpg")
    print(len(rfi_names), len(frb_names)) # expect (10371, 10944)

def load_data(prop=1):
    #TO DO
    return null

#functions for processing data

def get_label(file_path):               #True for frb, False for rfi
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == 'label_1'

def crop_1(img):                        #Crops the de-dispersed frequency time graph for first type of graphs
  return tf.image.crop_to_bounding_box(img, 369, 86, 268, 359)

def crop_2(img):                        #Crops the de-dispersed frequency time graph for the second type graph
  return tf.image.crop_to_bounding_box(img, 337, 83, 245, 351) 

def decode_img(img, height=244,width=244):   #reads the compressed file path and creates an image tensor
  img = tf.image.decode_jpeg(img, channels=3) # convert the compressed string to a 3D uint8 tensor
  img = tf.image.convert_image_dtype(img, tf.float32)
  if img.shape[0] == 689:                   #crop
    img = crop_1(img)
  else:
    img = crop_2(img)
  img = tf.image.rgb_to_grayscale(img)      #greyscale
  img = tf.image.resize(img, [height,width])#resize
  return img

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)      #load the raw data from the file as a string
  img = decode_img(img)
  return img, label


def visualize(image):                   #convenient function for plotting
  plt.imshow(tf.squeeze(image), vmin=0, vmax=1, cmap='Greys')


if __name__ == "__main__":
    main()