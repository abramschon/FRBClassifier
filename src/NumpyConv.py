import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    #read in example image
    image = np.asarray(Image.open("brams_website.png"))
    #display image
    #plot_im(image)
    print(f'Image dimensions {image.shape}')

    h_layer = np.vstack((np.zeros(3), np.ones(3), np.zeros(3))) #horizontal lines
    v_layer = np.transpose(h_layer)
    h_filter= np.dstack((h_layer,h_layer,h_layer,h_layer))
    v_filter = np.dstack((v_layer,v_layer,v_layer,v_layer))

    horisontal = convolve(image, h_filter)
    plot_im(horisontal)

    vertical = convolve(image, v_filter)
    plot_im(vertical)

def plot_im(image):
    plt.imshow(image)   
    plt.show()

def convolve(image, filter):
    #use no padding and stride of 1
    im_h, im_w, im_c = image.shape
    print(im_h, im_w)
    f_h, f_w, f_c = filter.shape

    output = np.zeros((im_h-f_h+1, im_w-f_w+1))

    for row in range(im_h-f_h+1):
        for col in range(im_w-f_w+1):
            output[row,col] = np.sum( image[row:row+f_h, col:col+f_w,:]*filter )

    return(output)

if __name__ == "__main__":
    main()