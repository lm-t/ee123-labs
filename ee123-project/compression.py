#initialise variables and import dependencies
#%pylab
import numpy as np
from numpy import *
#import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import misc
from scipy import ndimage
#from __future__ import division
from sys import getsizeof
import IPython
import scipy
# Import functions and libraries
#import numpy as np
#import matplotlib.pyplot as plt
import scipy
from PIL import Image

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
import matplotlib.pylab as pylab
import os
import imageio
# import zigzag functions
from zigzag import *


def formatImage(image):
    #img_color = image.resize(size, 1)
    img = np.array(image, dtype=np.float)
    return img

def reformatImage(image):
    img = np.array(image, dtype = uint8)
    return img
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def EE123_psnr(ref, meas, maxVal=255):
    assert np.shape(ref) == np.shape(meas), "Test image must match measured image dimensions"


    dif = (ref.astype(float)-meas.astype(float)).ravel()
    mse = np.linalg.norm(dif)**2/np.prod(np.shape(ref))
    psnr = 10*np.log10(maxVal**2.0/mse)
    return psnr

def rgb_to_ycbcr(r, g, b):
    y  = 0.299*r + 0.587*g    + 0.114*b
    cb = 128     - 0.168736*r - 0.331364*g + 0.5*b
    cr = 128     + 0.5*r      - 0.418688*g - 0.081312*b
    return y,cb,cr

def ycbcr_to_rgb(y,cb,cr):
    r = y + 1.4 * (cr - 128)
    g = y + (-0.343)*(cb - 128) + (-0.711)*(cr - 128)
    b = y + 1.765*(cb - 128)
    r[r > 255] = 255
    r[r < 0] = 0
    g[g > 255] = 255
    g[g < 0] = 0
    b[b > 255] = 255
    b[b < 0] = 0
    return r,g,b
quality=10
if quality <50:
    alpha=50/quality
else:  alpha=2-quality/50
# Quantization Matrix
QUANTIZATION_MAT_Y = alpha * np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
QUANTIZATION_MAT_C = alpha * np.array([[17, 18, 24, 47, 99, 99, 99, 99],[18, 21, 26, 66, 99, 99, 99, 99],[24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],])


def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream
# defining block size
block_size = 8


def compress(rgbImage,quality):
    global QUANTIZATION_MAT_C,QUANTIZATION_MAT_Y
    if quality <50:
        alpha=50/quality
    else:  alpha=2-quality/50
    # Quantization Matrix
    QUANTIZATION_MAT_Y = alpha * np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
    QUANTIZATION_MAT_C = alpha * np.array([[17, 18, 24, 47, 99, 99, 99, 99],[18, 21, 26, 66, 99, 99, 99, 99],[24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],])
    #Extract color channels
    redChannel = rgbImage[:,:,0]
    greenChannel = rgbImage[:,:,1]
    blueChannel = rgbImage[:,:,2]
    row,column,color = shape(rgbImage)

    #Calculating YCbCr values
    r=redChannel
    g=greenChannel
    b=blueChannel

    y,cb,cr = rgb_to_ycbcr(r,g,b)

    #8x8 Blockwise DCT on y, Cb and Cr
    #DCT on y
    imgy=y
    #size of y
    hy=imgy.shape[0]
    wy=imgy.shape[1]

    height_y = hy
    width_y = wy
    hy = np.float32(hy)
    wy = np.float32(wy)
    #No. of blocks to perform dct
    nbhy = math.ceil(hy/block_size)
    nbhy = np.int32(nbhy)

    nbwy = math.ceil(wy/block_size)
    nbwy = np.int32(nbwy)

    # height of padded image
    Hy =  block_size * nbhy

    # width of padded image
    Wy =  block_size * nbwy

    # create a numpy zero matrix with size of H,W
    padded_imgy = np.zeros((Hy,Wy))

    padded_imgy[0:height_y,0:width_y] = imgy[0:height_y,0:width_y]

    #the good stuff
    for i in range(nbhy):
            # Compute start and end row index of the block
            row_ind_1 = i*block_size
            row_ind_2 = row_ind_1+block_size

            for j in range(nbwy):

                # Compute start & end column index of the block
                col_ind_1 = j*block_size
                col_ind_2 = col_ind_1+block_size

                blocky = padded_imgy[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]

                # apply 2D discrete cosine transform to the selected block
                DCTy = dct2(blocky)

                DCT_normalizedy = np.divide(DCTy,QUANTIZATION_MAT_Y).astype(int)

                # reorder DCT coefficients in zig zag order by calling zigzag function
                # it will give you a one dimentional array
                reorderedy = zigzag(DCT_normalizedy)

                # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
                reshapedy= np.reshape(reorderedy, (block_size, block_size))

                # copy reshaped matrix into padded_img on current block corresponding indices
                padded_imgy[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedy

    #misc.imshow(np.uint8(padded_imgy))
    arrangedy = padded_imgy.flatten()


    bitstreamy = get_run_length_encoding(arrangedy)

    # Two terms are assigned for size as well, semicolon denotes end of image to reciever
    bitstreamy = str(padded_imgy.shape[0]) + " " + str(padded_imgy.shape[1]) + " " + bitstreamy + ";"

    #save as .txt
    filey = open("imagey.txt","w")
    filey.write(bitstreamy)
    filey.close()

    #DCT on cr
    imgcr=cr
    #size of cr
    hcr=imgcr.shape[0]
    wcr=imgcr.shape[1]

    height_cr = hcr
    width_cr = wcr
    hcr = np.float32(hcr)
    wcr = np.float32(wcr)
    #No. of blocks to perform dct
    nbhcr = math.ceil(hcr/block_size)
    nbhcr = np.int32(nbhcr)

    nbwcr = math.ceil(wcr/block_size)
    nbwcr = np.int32(nbwcr)

    # height of padded image
    Hcr =  block_size * nbhcr

    # width of padded image
    Wcr =  block_size * nbwcr

    # create a numpy zero matrix with size of H,W
    padded_imgcr = np.zeros((Hcr,Wcr))

    padded_imgcr[0:height_cr,0:width_cr] = imgcr[0:height_cr,0:width_cr]

    #the good stuff
    for i in range(nbhcr):
            # Compute start and end row index of the block
            row_ind_1 = i*block_size
            row_ind_2 = row_ind_1+block_size

            for j in range(nbwcr):

                # Compute start & end column index of the block
                col_ind_1 = j*block_size
                col_ind_2 = col_ind_1+block_size

                blockcr = padded_imgcr[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]

                # apply 2D discrete cosine transform to the selected block
                DCTcr = dct2(blockcr)

                DCT_normalizedcr = np.divide(DCTcr,QUANTIZATION_MAT_C).astype(int)

                # reorder DCT coefficients in zig zag order by calling zigzag function
                # it will give you a one dimentional array
                reorderedcr = zigzag(DCT_normalizedcr)

                # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
                reshapedcr= np.reshape(reorderedcr, (block_size, block_size))

                # copy reshaped matrix into padded_img on current block corresponding indices
                padded_imgcr[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedcr

    #misc.imshow(np.uint8(padded_imgy))
    arrangedcr = padded_imgcr.flatten()


    bitstreamcr = get_run_length_encoding(arrangedcr)

    # Two terms are assigned for size as well, semicolon denotes end of image to reciever
    bitstreamcr = str(padded_imgcr.shape[0]) + " " + str(padded_imgcr.shape[1]) + " " + bitstreamcr + ";"

    #save as .txt
    filecr = open("imagecr.txt","w")
    filecr.write(bitstreamcr)
    filecr.close()

    #8x8 Blockwise DCT on y, Cb and Cr
    #DCT on cb
    imgcb=cb
    #size of cb
    hcb=imgcb.shape[0]
    wcb=imgcb.shape[1]

    height_cb = hcb
    width_cb = wcb
    hcb = np.float32(hcb)
    wcb = np.float32(wcb)
    #No. of blocks to perform dct
    nbhcb = math.ceil(hcb/block_size)
    nbhcb = np.int32(nbhcb)

    nbwcb = math.ceil(wcb/block_size)
    nbwcb = np.int32(nbwcb)

    # height of padded image
    Hcb =  block_size * nbhcb

    # width of padded image
    Wcb =  block_size * nbwcb

    # create a numpy zero matrix with size of H,W
    padded_imgcb = np.zeros((Hcb,Wcb))

    padded_imgcb[0:height_cb,0:width_cb] = imgcb[0:height_cb,0:width_cb]

    #the good stuff
    for i in range(nbhcb):
            # Compute start and end row index of the block
            row_ind_1 = i*block_size
            row_ind_2 = row_ind_1+block_size

            for j in range(nbwcb):

                # Compute start & end column index of the block
                col_ind_1 = j*block_size
                col_ind_2 = col_ind_1+block_size

                blockcb = padded_imgcb[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]

                # apply 2D discrete cosine transform to the selected block
                DCTcb = dct2(blockcb)

                DCT_normalizedcb = np.divide(DCTcb,QUANTIZATION_MAT_C).astype(int)

                # reorder DCT coefficients in zig zag order by calling zigzag function
                # it will give you a one dimentional array
                reorderedcb = zigzag(DCT_normalizedcb)

                # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
                reshapedcb= np.reshape(reorderedcb, (block_size, block_size))

                # copy reshaped matrix into padded_img on current block corresponding indices
                padded_imgcb[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedcb

    #misc.imshow(np.uint8(padded_imgy))
    arrangedcb = padded_imgcb.flatten()


    bitstreamcb = get_run_length_encoding(arrangedcb)

    # Two terms are assigned for size as well, semicolon denotes end of image to reciever
    bitstreamcb = str(padded_imgcb.shape[0]) + " " + str(padded_imgcb.shape[1]) + " " + bitstreamcb + ";"


    #save as .txt
    filecb = open("imagecb.txt","w")
    filecb.write(bitstreamcb)
    filecb.close()
    return bitstreamy , bitstreamcr, bitstreamcb


def decompress(bitstreamy,bitstreamcr,bitstreamcb):
    global QUANTIZATION_MAT_C,QUANTIZATION_MAT_Y
    decodeimg_y=bitstreamy
    decodeimg_cr=bitstreamcr
    decodeimg_cb=bitstreamcb
    #Decodes y
    #splits into tokens seperated by space characters
    details_y = decodeimg_y.split()

    # just python-crap to get integer from tokens : h and w are height and width of image (first two items)
    hy = int(''.join(filter(str.isdigit, details_y[0])))
    wy = int(''.join(filter(str.isdigit, details_y[1])))

    # declare an array of zeros (It helps to reconstruct bigger array on which IDCT and all has to be applied)
    array_y = np.zeros(hy*wy).astype(int)


    # some loop var initialisation
    k = 0
    i = 2
    x = 0
    j = 0


    # This loop gives us reconstructed array of size of image

    while k < array_y.shape[0]:
    # Oh! image has ended
        if(details_y[i] == ';'):
            break
    # This is imp! note that to get negative numbers in array check for - sign in string
        if "-" not in details_y[i]:
            array_y[k] = int(''.join(filter(str.isdigit, details_y[i])))
        else:
            array_y[k] = -1*int(''.join(filter(str.isdigit, details_y[i])))

        if(i+3 < len(details_y)):
            j = int(''.join(filter(str.isdigit, details_y[i+3])))

        if j == 0:
            k = k + 1
        else:
            k = k + j + 1

        i = i + 2

    array_y = np.reshape(array_y,(hy,wy))

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    j = 0
    k = 0

    # initialisation of compressed image
    padded_imgy = np.zeros((hy,wy))

    while i < hy:
        j = 0
        while j < wy:
            temp_streamy = array_y[i:i+8,j:j+8]
            blocky = inverse_zigzag(temp_streamy.flatten(), int(block_size),int(block_size))
            de_quantizedy = np.multiply(blocky,QUANTIZATION_MAT_Y)
            padded_imgy[i:i+8,j:j+8] = idct2(de_quantizedy)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    padded_imgy[padded_imgy > 255] = 255
    padded_imgy[padded_imgy < 0] = 0

    #Decodes cb
    # spplits into tokens seperated by space characters
    details_cb = decodeimg_cb.split()

    # just python-crap to get integer from tokens : h and w are height and width of image (first two items)
    hcb = int(''.join(filter(str.isdigit, details_cb[0])))
    wcb = int(''.join(filter(str.isdigit, details_cb[1])))

    # declare an array of zeros (It helps to reconstruct bigger array on which IDCT and all has to be applied)
    array_cb = np.zeros(hcb*wcb).astype(int)


    # some loop var initialisation
    k = 0
    i = 2
    x = 0
    j = 0


    # This loop gives us reconstructed array of size of image

    while k < array_cb.shape[0]:
    # Oh! image has ended
        if(details_cb[i] == ';'):
            break
    # This is imp! note that to get negative numbers in array check for - sign in string
        if "-" not in details_cb[i]:
            array_cb[k] = int(''.join(filter(str.isdigit, details_cb[i])))
        else:
            array_cb[k] = -1*int(''.join(filter(str.isdigit, details_cb[i])))

        if(i+3 < len(details_cb)):
            j = int(''.join(filter(str.isdigit, details_cb[i+3])))

        if j == 0:
            k = k + 1
        else:
            k = k + j + 1

        i = i + 2

    array_cb = np.reshape(array_cb,(hcb,wcb))

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    j = 0
    k = 0

    # initialisation of compressed image
    padded_imgcb = np.zeros((hcb,wcb))

    while i < hcb:
        j = 0
        while j < wcb:
            temp_streamcb = array_cb[i:i+8,j:j+8]
            blockcb = inverse_zigzag(temp_streamcb.flatten(), int(block_size),int(block_size))
            de_quantizedcb = np.multiply(blockcb,QUANTIZATION_MAT_C)
            padded_imgcb[i:i+8,j:j+8] = idct2(de_quantizedcb)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    padded_imgcb[padded_imgcb > 255] = 255
    padded_imgcb[padded_imgcb < 0] = 0

    #Decodes cr

    # splits into tokens seperated by space characters
    details_cr = decodeimg_cr.split()

    # just python-crap to get integer from tokens : h and w are height and width of image (first two items)
    hcr = int(''.join(filter(str.isdigit, details_cr[0])))
    wcr = int(''.join(filter(str.isdigit, details_cr[1])))

    # declare an array of zeros (It helps to reconstruct bigger array on which IDCT and all has to be applied)
    array_cr = np.zeros(hcr*wcr).astype(int)


    # some loop var initialisation
    k = 0
    i = 2
    x = 0
    j = 0


    # This loop gives us reconstructed array of size of image

    while k < array_cr.shape[0]:
    # Oh! image has ended
        if(details_cr[i] == ';'):
            break
    # This is imp! note that to get negative numbers in array check for - sign in string
        if "-" not in details_cb[i]:
            array_cr[k] = int(''.join(filter(str.isdigit, details_cr[i])))
        else:
            array_cr[k] = -1*int(''.join(filter(str.isdigit, details_cr[i])))

        if(i+3 < len(details_cr)):
            j = int(''.join(filter(str.isdigit, details_cr[i+3])))

        if j == 0:
            k = k + 1
        else:
            k = k + j + 1

        i = i + 2

    array_cr = np.reshape(array_cr,(hcr,wcr))

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    j = 0
    k = 0

    # initialisation of compressed image
    padded_imgcr = np.zeros((hcr,wcr))

    while i < hcr:
        j = 0
        while j < wcr:
            temp_streamcr = array_cr[i:i+8,j:j+8]
            blockcr = inverse_zigzag(temp_streamcr.flatten(), int(block_size),int(block_size))
            de_quantizedcr = np.multiply(blockcr,QUANTIZATION_MAT_C)
            padded_imgcr[i:i+8,j:j+8] = idct2(de_quantizedcr)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    padded_imgcr[padded_imgcr > 255] = 255
    padded_imgcr[padded_imgcr < 0] = 0

    #Convert back to RGB
    y=padded_imgy
    cr=padded_imgcr
    cb=padded_imgcb

    r1,g1,b1 = ycbcr_to_rgb(y,cb,cr)

    #recombine RGB
    recombinedRGBImage1 = np.dstack((r1, g1, b1))
    recombinedRGBImage2=reformatImage(recombinedRGBImage1)

    return recombinedRGBImage2
