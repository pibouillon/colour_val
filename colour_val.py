 ##import the folowing libraries 
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage.color
import skimage.io 
from math import *

##visual display
def color_pixel_values(data,color_channel):
    n, bins, patches = plt.hist(data, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

    if color_channel == "L*" :
    
        n = n.astype('int') 
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.Greys(bins[i]/max(bins)))# Make one bin stand out   
        plt.title('Distribution of L* pixel values', fontsize=12)
        plt.xlabel('L values', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.show()


    elif color_channel == "a*" :
    
        n = n.astype('int') 
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.Reds(bins[i]/max(bins)))# Make one bin stand out   
        plt.title('Distribution of a* pixel values', fontsize=12)
        plt.xlabel('a* values', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.show()


    elif color_channel == "b*" :
    
        n = n.astype('int') 
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.YlGnBu(bins[i]/max(bins)))# Make one bin stand out   
        plt.title('Distribution of b* pixel values', fontsize=12)
        plt.xlabel('b* values', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.show()

    else : 
        print("please indicate a CIEL*a*b* color parameter")


color_pixel_values(L[L_no_bg],"L*")


def plot3d(image): 
    image_lab = skimage.color.rgb2lab(image)
    #3d plot of Lab values
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # data for three-dimensional scattered points
    zdata = image_lab[:, :, 1]#a                  
    xdata = image_lab[:, :, 0]#L
    ydata = image_lab[:, :, 2]#b
    ax.scatter3D(xdata, ydata, zdata,
                alpha = 0.8,c = (zdata+xdata+ydata))
    ax.set_xlabel('L channel')
    ax.set_ylabel('b* channel')
    ax.set_zlabel('a* channel')
    plt.show()

def show_lab(image):
    image_lab = skimage.color.rgb2lab(image)
    # Lab Color Space - https://en.wikipedia.org/wiki/CIELAB_color_space
    L = image_lab[:, :, 0]
    a = image_lab[:, :, 1]
    b = image_lab[:, :, 2]
    fig, ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(L)
    ax[1].imshow(a)
    ax[2].imshow(b)
    plt.show()



def analyse(image):
    #color conversion in the following color space : RGB, L*a*bù* and HSV.
    image_lab = skimage.color.rgb2lab(image)

    #extract single channel for RGB Color space
    R=image[:, :, 0]
    R_no_bg = R > 0
    #calculate mean and standard deviation for R channel
    R_mean=round(np.mean(R[R_no_bg].T),2)
    R_var=round(np.std(R[R_no_bg].T),2)

    G=image[:, :, 1]
    G_no_bg = G > 0
    #calculate mean and standard deviation for G channel
    G_mean=round(np.mean(G[G_no_bg].T),2)
    G_var=round(np.std(G[G_no_bg].T),2)

    B=image[:, :, 2]
    B_no_bg = B > 0
    #calculate mean and standard deviation for B channel
    B_mean=round(np.mean(B[B_no_bg].T),2)
    B_var=round(np.std(B[B_no_bg].T),2)
    
    L = image_lab[:, :, 0]
    L_no_bg = L > 0
    #calculate mean and standard deviation for L* channel
    L_mean=round(np.mean(L[L_no_bg]).T,2)
    L_var=round(np.std(L[L_no_bg]).T,2)
    #number of unique values in L
    #unique=np.unique(L)
    #num=np.shape(unique)
    a = image_lab[:, :, 1]
    a_no_bg = a != 0
    #calculate mean and standard deviation for a* channel
    a_mean=round(np.mean(a[a_no_bg]).T,2)
    a_var=round(np.std(a[a_no_bg]).T,2)

    #unique_a=np.unique(a)
    #num_a=np.shape(unique_a)
    b = image_lab[:, :, 2]
    b_no_bg = b != 0
    #calculate mean and standard deviation for b* channel
    b_mean=round(np.mean(b[b_no_bg]).T,2)
    b_var=round(np.std(b[b_no_bg]).T,2)
    #unique_b=np.unique(b)
    #num_b=np.shape(unique_b)

    chroma = sqrt(a_mean**2 + b_mean**2)
    hue = np.arctan(b_mean/a_mean)

    #compilation of color descriptors
    list_colour_results=(R_mean,G_mean,B_mean,L_mean,a_mean,b_mean,R_var,G_var,B_var,L_var,a_var,b_var, chroma, hue)
    return list_colour_results


#indicate path of .png files
path = r'your_path\images/'
image_path_list = os.listdir(r'your_path\images') # looking at the first image

##looping images in defined folder
i = 0
#for n in range (0,len(image_path_list)):
for i in range (0,len(image_path_list)):
    image_path = image_path_list[i]
    image = skimage.io.imread(r"your_path\images\"+ image_path)
    if i == 0 :
        with open(path+"resultats.txt", "a") as temp_file_result:
            temp_file_result.write("image,R_mean,G_mean,B_mean,L_mean,a_mean,b_mean,R_var,G_var,B_var,L_var,a_var,b_var,chroma,hue,\n")
        with open(path+"features.txt","a") as tem_file_peaks : 
            tem_file_peaks.write("image,peak(s),\n")
    #calculate colorimetric descriptors for each labelled regions (each fruit)
    res=analyse(image)
    with open(path+"resultats.txt", "a") as temp_file_result:
        temp_file_result.write(str(image_path)+","+str(res)+"\n")
    i=+1

