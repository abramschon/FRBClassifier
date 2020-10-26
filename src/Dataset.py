import numpy as np
from matplotlib import pyplot as plt
import os 
import random as rnd
import tensorflow as tf

def main():
  dataset = Dataset(0.1, [12,12]) #load 10% of the data
  train_ds, val_ds, test_ds = dataset.get_datasets() 

  #check out the first image
  for img, lab in train_ds.unbatch().take(1):
      visualize(img)
      print(f"Label: {lab}")

class Dataset:
  def __init__(self, prop=1, im_shape = [244,244],  batch_size=32,):
    #get file paths for the rfis and frbs
    rfi_names = tf.io.gfile.glob("data/label_0/*.jpg") #assumes testing from 'root' directiory
    frb_names = tf.io.gfile.glob("data/label_1/*.jpg")
    all_names = rfi_names + frb_names
    rnd.seed(17) #set seed important for reproducibility 
    rnd.shuffle(all_names) #shuffle names

    #define train, validation, test counts
    no_total = round(len(all_names)*prop) 
    self.no_train = round(0.7 * no_total)
    self.no_val = round(0.15 * no_total)
    self.no_test = no_total - self.no_train - self.no_val     

    #record steps per batch size
    self.steps_per_epoch = self.no_train // batch_size     

    # create datasets of the file names
    list_ds = tf.data.Dataset.from_tensor_slices(all_names)
    train_list_ds = list_ds.take(self.no_train)
    val_list_ds = list_ds.skip(self.no_train).take(self.no_val)
    test_list_ds = list_ds.skip(self.no_train + self.no_val).take(self.no_test)

    # create actual datasets
    self.im_shape = im_shape #shape of each image
    AUTOTUNE = tf.data.experimental.AUTOTUNE #read into this more
    process = lambda file: process_path(file, im_shape)
    train_ds = train_list_ds.map(process, num_parallel_calls=AUTOTUNE)
    val_ds = val_list_ds.map(process, num_parallel_calls=AUTOTUNE)
    test_ds = test_list_ds.map(process, num_parallel_calls=AUTOTUNE)

    #configure to be batched, etc.
    self.train_ds = configure_for_performance(train_ds, batch_size, train=True)
    self.val_ds = configure_for_performance(val_ds, batch_size)
    self.test_ds = configure_for_performance(test_ds, batch_size)

  def get_datasets(self):
    return (self.train_ds, self.val_ds, self.test_ds)



def configure_for_performance(ds, batch_size=32, train=False): #cache, batch and prefetch data
  ds = ds.cache() 
  if train:
    #apparently best practise in Keras is to repeat then batch for training
    ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # allows later elements to be prepared while the current element is being processed
  return ds

#======================================functions for processing images
def illustration(): #illustrates the process
  rfi_names = tf.io.gfile.glob("data/label_0/*.jpg") #assumes testing from 'root' directiory
  frb_names = tf.io.gfile.glob("data/label_1/*.jpg")
  print(len(rfi_names), len(frb_names)) # expect (10371, 10944)

  ex_path = tf.io.read_file(frb_names[10000])
  img = tf.image.decode_jpeg(ex_path, channels=3) # convert the compressed string to a 3D uint8 tensor
  print(f"Dimensions {img.shape}")
  print(img[1,1,:].numpy()) #inspect the first pixel

  #convert the uint8 tensor  to float32
  img = tf.image.convert_image_dtype(img, tf.float32) #this also has the effect of squishing all pixel values to [0,1]
  print(img[1,1,:].numpy()) #inspect the first pixel
  
  #convert to grayscale
  img = tf.image.rgb_to_grayscale(img)  
  print(img[1,1,:].numpy()) #inspect the first pixel - notice the collapse along the frequency channels

  #crop based on size
  if img.shape[0] == 689:                 
      img = crop_1(img)
  else:
      img = crop_2(img)
  
  visualize(img)


def get_label(file_path):               #True for frb, False for rfi
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == 'label_1'

def crop_1(img):                        #Crops the de-dispersed frequency time graph for first type of graphs
  return tf.image.crop_to_bounding_box(img, 369, 86, 268, 359)

def crop_2(img):                        #Crops the de-dispersed frequency time graph for the second type graph
  return tf.image.crop_to_bounding_box(img, 337, 83, 245, 351) 

def decode_img(img, shape=[244,244]):   #reads the compressed file path and creates an image tensor
  img = tf.image.decode_jpeg(img, channels=3) # convert the compressed string to a 3D uint8 tensor
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.cond( tf.shape(img)[0] == 689, lambda: crop_1(img), lambda: crop_2(img) ) #crop
  img = tf.image.rgb_to_grayscale(img)      #greyscale
  img = tf.image.resize(img, shape)#resize
  return img

def process_path(file_path, shape=[244,244]):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)      #load the raw data from the file as a string
  img = decode_img(img, shape)
  return img, label

def visualize(image):                   #convenient function for plotting
  plt.imshow(tf.squeeze(image), vmin=0, vmax=1, cmap='Greys')
  plt.show()


if __name__ == "__main__":
    main()