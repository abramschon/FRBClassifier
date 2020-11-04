import numpy as np
import matplotlib.pyplot as plt
from NumpyConv import image_to_ndarray, plot_im

def main():
    #we are going to use a 2x2 filter with stride 2
    pool_dim = (2,2)
    stride = 2

    #read in example images and plot them, and plot their down-sampled counter parts
    files = ["../example_data/L.png" ,          #image of an L 
            "../example_data/FRB.jpg",          #de-dispersed freq-time plot FRB from this paper  
            "../example_data/FRBsmol.jpeg",          #scaled down de-dispersed freq-time plot FRB from this paper  
             "../example_data/FETCHRFI.png"]    #DM plot for RFI from FETCH paper (Agarwal et al. 2020)

    for f in files:
        image = image_to_ndarray(f)
        plot_im(image)                                  #plot original image
        print(f'Image dimensions {image.shape}')                 
        
        #first round of max pooling
        pooled_im = max_pool(image, pool_dim, stride)   #apply max pooling
        plot_im(pooled_im)                              #plot down-sampled image
        print(f'1st maxpool image dimensions {pooled_im.shape}')

        #second round of max pooling
        pooled_im = max_pool(pooled_im, pool_dim, stride)   #apply max pooling
        plot_im(pooled_im)                              #plot down-sampled image
        print(f'2nd maxpool image dimensions {pooled_im.shape}')



def max_pool(image, pool_dim, stride):
    im_h, im_w, im_c = image.shape
    p_h, p_w = pool_dim
    
    #work out output dimensions (chanels are preserved)
    output_h = int((im_h-p_h)/stride + 1)
    output_w = int((im_w-p_w)/stride + 1)

    output = np.zeros((output_h, output_w, im_c), dtype=int)

    for row in range(output_h):
        for col in range(output_w):
            for depth in range(im_c):
                output[row,col,depth] = np.max( image[row*stride:row*stride+p_h, col*stride:col*stride+p_w, depth] )
                
    print(output.shape)
    return(output)


if __name__ == "__main__":
    main()   

