import numpy as np
import os 
import tensorflow as tf

def main():
  test = process_images("data/label_0", prop=0.1)

  #check out the first image
  for img, id in test.unbatch().take(1):
      print(f"Id: {id}")

def process_images(dir="data", prop=1, im_shape = [224,224], mean=0, sd=1):
  """
    Takes in a directory of images and returns a tensorflow dataset or id image pairs.
  """

  #get file names 
  names = tf.io.gfile.glob(f"{dir}/*.jpg")      
  n = round(len(names)*prop) # number of observations used

  # create datasets of the file names
  list_ds = tf.data.Dataset.from_tensor_slices(names[:n])

  # process dataset
  AUTOTUNE = tf.data.experimental.AUTOTUNE 

  process = lambda file: process_path(file, im_shape, mean, sd)
  ds = list_ds.map(process, num_parallel_calls=AUTOTUNE)
  ds = ds.batch(n) #model expects data to be batched

  return ds

#======================================functions for processing images

def crop_1(img): 
  """ Crops the de-dispersed frequency time graph for 1st type of graph """
  return tf.image.crop_to_bounding_box(img, 369, 86, 268, 359)

def crop_2(img): 
  """ Crops the de-dispersed frequency time graph for the 2nd type graph """
  return tf.image.crop_to_bounding_box(img, 337, 83, 245, 351) 

def decode_img(img, shape=[244,244], mean=0, sd=1):   
  """
    Reads the file path and processes it to create an image tensor
  """
  img = tf.image.decode_jpeg(img, channels=3) # convert the compressed string to a 3D uint8 tensor
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = (img - mean) / sd #normalise
  img = tf.cond( tf.shape(img)[0] == 689, lambda: crop_1(img), lambda: crop_2(img) ) #crop
  img = tf.image.rgb_to_grayscale(img) #greyscale
  img = tf.image.resize(img, shape) #resize
  return img

def get_id(file_path):       
  """
    Returns the name of the file as its id
  """       
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-1] 

def process_path(file_path, shape=[224,242], mean=0, sd=1):
  """
    Takes in the file paths and returns id, image pairs using tf data types
  """
  label = get_id(file_path)
  img = tf.io.read_file(file_path)      
  img = decode_img(img, shape, mean, sd)
  return img, label 

if __name__ == "__main__":
    main()
