import numpy as np
import matplotlib.pyplot as plt
from NumpyConv import image_to_ndarray, plot_im
import tensorflow as tf

def main():

    #read in example images and plot them, and plot their down-sampled counter parts
    files = ["../example_data/L.png" ,          #image of an L 
            "../example_data/FRB.jpg",          #de-dispersed freq-time plot FRB from this paper  
             "../example_data/FETCHRFI.png"]    #DM plot for RFI from FETCH paper (Agarwal et al. 2020)
    
    for f in files:
        image = image_to_ndarray(f)
        #plot_im(image)                          #plot original image
        print(f'Image dimensions {image.shape}')                 

        g_a_p = global_average_pool(image)      #apply global average pooling
        print("Global average pooling:", g_a_p)

        tensor = tf.constant(image)             #compare with keras implementation
        tensor = tf.expand_dims(tensor, axis=0) #keras expects batched data
        keras_g_a_p = tf.keras.layers.GlobalAveragePooling2D()(tensor)
        print("Keras G.A.P", keras_g_a_p.numpy()[0]) 

def global_average_pool(image):
    im_h, im_w, im_c = image.shape

    output = np.zeros(im_c, dtype=float)

    for channel in range(im_c):
        output[channel] = np.mean(image[:,:,channel])
   
    return(output)


if __name__ == "__main__":
    main()   

