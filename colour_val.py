 ##import the folowing libraries 
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.api as sm


##visual display
image=cv2.imread(r".png")
show_lab(image)
plot3d(image)
get_peak(image)


def plot3d(image): 
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
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
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)   
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #extract single channel for RGB Color space
    R=image_rgb[:, :, 0]
    R_no_bg = R > 0
    #calculate mean and standard deviation for R channel
    R_mean=round(np.mean(R[R_no_bg].T),2)
    R_var=round(np.std(R[R_no_bg].T),2)
    G=image_rgb[:, :, 1]
    G_no_bg = G > 0
    #calculate mean and standard deviation for G channel
    G_mean=round(np.mean(G[G_no_bg].T),2)
    G_var=round(np.std(G[G_no_bg].T),2)
    B=image_rgb[:, :, 2]
    B_no_bg = B > 0
    #calculate mean and standard deviation for B channel
    B_mean=round(np.mean(B[B_no_bg].T),2)
    B_var=round(np.std(B[B_no_bg].T),2)
    
    
    #extract single channel for L*a*b* color space
    #The LAB values returned from OpenCV will never lie outside the ranges 0 ≤ L ≤ 100, 
    # -127 ≤ a ≤ 127, -127 ≤ b ≤ 127 
    # when converting float images (OpenCV color conversions). 
    # When converting 8-bit images, L -> L * 255/100, 
    # and a and b -> a and b + 128 to fill out the 8-bit range.
    L = image_lab[:, :, 0]
    L_no_bg = L > 0
    L=L[L_no_bg]
    #calculate mean and standard deviation for L* channel
    L_mean=round(np.mean(L).T,2)
    L_var=round(np.std(L).T,2)
    #number of unique values in L
    #unique=np.unique(L)
    #num=np.shape(unique)
    a = image_lab[:, :, 1]
    a_no_bg = a != 128
    #calculate mean and standard deviation for a* channel
    a_mean=round(np.mean(a).T,2)
    a_var=round(np.std(a).T,2)
    #unique_a=np.unique(a)
    #num_a=np.shape(unique_a)
    b = image_lab[:, :, 2]
    b_no_bg = b != 128
    #calculate mean and standard deviation for b* channel
    b_mean=round(np.mean(b).T,2)
    b_var=round(np.std(b).T,2)
    #unique_b=np.unique(b)
    #num_b=np.shape(unique_b)

    #extract single channel for HSV color space
    #during color conversion -> V = max(R,G,B)
    H = image_hsv[:, :, 0]
    H_no_bg = H > 0
    #calculate mean and standard deviation for H channel
    H_mean=round(np.mean(H[H_no_bg].T)*2,2)
    H_var=round(np.std(H[H_no_bg].T)*2,2)
    S = image_hsv[:, :, 1]
    S_no_bg = S > 0
    #calculate mean and standard deviation for S channel
    S_mean=round(np.mean(S[S_no_bg].T),2)
    S_var=round(np.std(S[S_no_bg].T),2)
    V = image_hsv[:, :, 2]
    V_no_bg = V > 0
    #calculate mean and standard deviation for V channel
    V_mean=round(np.mean(V[V_no_bg].T),2)
    V_var=round(np.std(V[V_no_bg].T),2)

    #compilation of color descriptors
    list_colour_results=(R_mean,G_mean,B_mean,L_mean,a_mean,b_mean,H_mean,S_mean,V_mean,R_var,G_var,B_var,L_var,a_var,b_var,H_var,S_var,V_var)
    return list_colour_results

def get_peak(image):

    #Our initial axiom is that we can describe pattern formation in apple according to at least one morphogen influencing red flesh colour. 
    # So we can describe pixel values density according to the colouration pattern. 
    #Pixel values can be approximated by a Gaussian distribution called A related to the existence of one morphogen
    #With this approximation we can define two colour distribution scenarios : 
    # (i) a scattered flesh colour approximated by a Gaussian distribution with a unique value of µ corresponding 
    # to a predominant colour that could be independently white, yellow or red and imply the existence of one unique morphogen, 
    # and (ii) a segmented distribution with separated non-red and red flesh parts, pixel values density 
    # can be approximated by a bimodal distribution implying two morphogens
    #with A the pixel values density describing the non-red part of the flesh, 
    # µ1 giving the non-red predominant colour; and B the pixel values density of the red part of the flesh, 
    # µ2 giving the red predominant colour.

    #visual display in comment 

    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    #conversion from BGR to L*a*b*
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #pixel distribution of a* values
    a_fit = image_lab[:, :, 1]
    a_fit=a_fit[a_fit>128]

    #frequency and bins from the histogram
    frequency, bins = np.histogram(a_fit, bins=127, range=[128, 255])
    bins=np.delete(bins,0)   
  
    #density estimation for pixel distribution
    kde = sm.nonparametric.KDEUnivariate(a_fit)
    kde.fit(bw=2.5)  # Estimate the densities
    #plt.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)
    #plt.show()

    #peak(s) detection 
    peaks, _ = signal.find_peaks(kde.density,distance=1000
                                ,height=max(kde.density)/10)
    #prominence(s) of peak(s)
    prominences = signal.peak_prominences(kde.density, peaks)[0]
    #plt.plot(kde.density)
    #plt.plot(peaks, kde.density[peaks], "x")
    #plt.plot(np.zeros_like(kde.density), "--", color="gray")
    #plt.show()

    #return mod(s) of detected peak(s)
    mods = kde.support[peaks]

    #create a dictionnary with {mods:prominences}
    predom = {}
    for i,j in zip(mods,prominences):
        predom[i] = j
    #we define less than two predominant colors : one for a scattered distribution; 
    #two for a segmeented distribution with coexistence of red-flesh and white-flesh parts
    if len (predom)>2:
        print (predom)
        mod=list(predom)[0],list(predom)[-1]
        prominence=list(predom.values())[0],list(predom.values())[-1]
        print(mod)
        print(prominence)
        return mod,prominence
    else:
        return mods,prominences


#indicate path of .png files
path = r'/images/'
image_path_list = os.listdir(r'/images') # looking at the first image

##looping images in defined folder
i = 0
#for n in range (0,len(image_path_list)):
for i in range (0,len(image_path_list)):
    image_path = image_path_list[i]
    image = cv2.imread(r"C/images/"+ image_path)
    if i == 0 :
        with open(path+"resultats.txt", "a") as temp_file_result:
            temp_file_result.write("image,R_mean,G_mean,B_mean,L_mean,a_mean,b_mean,H_mean,S_mean,V_mean,R_var,G_var,B_var,L_var,a_var,b_var,H_var,S_var,V_var,\n")
        with open(path+"features.txt","a") as tem_file_peaks : 
            tem_file_peaks.write("image,peak(s),\n")
    #calculate colorimetric descriptors for each labelled regions (each fruit)
    res=analyse(image)
    peaks,prominences=get_peak(image)
    with open(path+"resultats.txt", "a") as temp_file_result:
        temp_file_result.write(str(image_path)+","+str(res)+"\n")
    with open(path+"features.txt","a") as tem_file_peaks : 
            tem_file_peaks.write(str(image_path)+","+str(peaks)+","+str(prominences)+"\n") 
    i=+1


