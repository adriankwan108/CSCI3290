import argparse
import numpy as np
import imageio

PI = 3.14

def imread(path):
    """
    :param path: image path to read, str format
    :return: image data in ndarray format, the scale for the image is from 0.0 to 1.0
    """
    assert isinstance(path, str), 'Please use str as your path!'
    assert (path[-3:] == 'png') or (path[-3:] == 'PNG'), 'This assignment only support PNG grayscale images!'
    im = imageio.imread(path)
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = im / 255.
    return im

def imwrite(im, path):
    """
    :param im: image to save, ndarray format, the scale for the image is from 0.0 to 1.0
    :param path: path to save the image, str format
    """
    assert isinstance(im, np.ndarray), 'Please use ndarray data structure for your image to save!'
    assert isinstance(path, str), 'Please use str as your path!'
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = (im * 255.0).astype(np.uint8)
    imageio.imwrite(path, im)

def gaussian_kernel(size, sigma):
    """
    :param size: kernel size: size x size, int format
    :param sigma: standard deviation for gaussian kernel, float format
    :return: gaussian kernel in ndarray format
    """
    assert isinstance(size, int), 'Please use int for the kernel size!'
    assert isinstance(sigma, float), 'Please use float for sigma!'
    
    # ##################### Implement this function here ##################### #
    kernel = np.zeros(shape=[size, size], dtype=float)  # this line can be modified
    kernel_max = size//2+1
    kernel_min = 0-kernel_max+1
    y,x = np.mgrid[kernel_min:kernel_max , kernel_min:kernel_max]

    kernel = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel = kernel / kernel.sum() #normalized
    # ######################################################################## #
    assert isinstance(kernel, np.ndarray), 'please use ndarray as you kernel data format!'
    return kernel

def conv(im_in, kernel):
    """
    :param im_in: image to be convolved, ndarray format
    :param kernel: kernel use to convolve, ndarray format
    :return: result image, ndarray format
    """
    assert isinstance(im_in, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(kernel, np.ndarray), 'Please use ndarray data structure for your kernel!'

    # ##################### Implement this function here ##################### #

    kr = int(np.sqrt(kernel.size)) #kernel range
    ir = int(np.sqrt(im_in.size))  #im range

    size = ir-kr+1
    im_out = np.zeros(shape=[size,size], dtype=float)
    
    for kx in range(size):
        for ky in range(size):
            temp = 0
            for i in range(kr): #for each box in kernel
                for j in range(kr):
                    temp = temp + kernel[i][j] * im_in[kx+i][ky+j] #multiply and sum up
            im_out[kx][ky] = temp

    return im_out
    # ######################################################################## #


def sharpen(im_input, im_smoothed):
    """
    :param im_input: the original image, ndarray format
    :param im_smoothed: the smoothed image, ndarray format
    :return: sharoened image, ndarray format
    """
    assert isinstance(im_input, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(im_smoothed, np.ndarray), 'Please use ndarray data structure for your image!'

    # ##################### Implement this function here ##################### #
    input_size = int(np.sqrt(im_input.size))
    crop_size = (int(input_size-np.sqrt(im_smoothed.size)))//2
    
    crop_image = im_input[crop_size:-crop_size, crop_size:-crop_size]
    detail_image = np.subtract(crop_image, im_smoothed)

    sharp_image =  np.add(crop_image , detail_image)

    return sharp_image
    # ######################################################################## #

def main():
    parser = argparse.ArgumentParser(description='Image Sharpening')
    parser.add_argument('--input', type=str, default='test_01.png', help='path of the input image')
    parser.add_argument('--kernel', type=int, default=5, help='the square kernel size')
    parser.add_argument('--sigma', type=float, default=1.5, help='the standard deviation in gaussian kernel')
    parser.add_argument('--output', type=str, default='output_01.png', help='the path of the output image')
    args = parser.parse_args()

    im = imread(args.input)
    kernel = gaussian_kernel(size=args.kernel, sigma=args.sigma)
    smoothed_im = conv(im_in=im, kernel=kernel)
    sharpened_im = sharpen(im_input=im, im_smoothed=smoothed_im)
    imwrite(im=sharpened_im, path=args.output)

if __name__ == '__main__':
    main()
