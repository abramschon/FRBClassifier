import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    #read in example image
    rfi_file = "../example_data/FETCHRFI.png"
    l_file = "../example_data/L.png"

    image = np.asarray(Image.open(rfi_file))
    #display image
    plot_im(image)
    shape = image.shape
    print(f'Image dimensions {image.shape}')
    
    #create a simple filters:
    # "Vertical"    "Horizontal"
    #   0 1 0         0 0 0
    #   0 1 0   and   1 1 1
    #   0 1 0         0 0 0

    h_layer = np.vstack((np.zeros(3), np.ones(3), np.zeros(3))) #horizontal filter
    v_layer = np.transpose(h_layer)                             #vertical filter
    h_filter= np.dstack([h_layer for i in range(shape[2])])     #stack along depth channel
    v_filter = np.dstack([v_layer for i in range(shape[2])])

    #apply filters to image
    horisontal = convolve(image, h_filter)
    plot_im(horisontal)

    vertical = convolve(image, v_filter)
    plot_im(vertical)

    print(f'Output image dimensions {horisontal.shape}')

def plot_im(image, bw=True):
    if bw:
        plt.imshow(image, cmap=plt.cm.get_cmap('Greys').reversed())   
    else:
        plt.imshow(image)

    plt.show()

def convolve(image, filter):
    #use no padding and stride of 1
    im_h, im_w, im_c = image.shape
    f_h, f_w, f_c = filter.shape

    output = np.zeros((im_h-f_h+1, im_w-f_w+1))

    for row in range(im_h-f_h+1):
        for col in range(im_w-f_w+1):
            output[row,col] = np.sum( image[row:row+f_h, col:col+f_w,:]*filter )

    return(output)

if __name__ == "__main__":
    main()