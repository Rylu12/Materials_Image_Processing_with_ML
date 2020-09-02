import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

def k_cluster(img, k=3, max_iter=10, epsilon=1):
    """
    Function to perform K-Means Clustering on images using OpenCV

    :param img: Source of image
    :param k: Number of color clusters
    :param max_iter: Stopping Criteria (Maximum iterations to run K-Means Clustering)
    :param epsilon: Stopping Criteria (Specified accuracy to stop the iterations when reached)
    :return: New clustered image
    """


    img2 = img.reshape((-1, 3))
    img2 = np.float32(img2)

    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    attempts = 100

    ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    img = res.reshape((img.shape))

    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap='gray')
    plt.show()
    return img


def get_features(img):
    """
    Function to get 62 feature variables (Pixel, Gabor Filters, Edge Detectors, Gaussian Filters, and Median Filters) for machine learning training
    
    :param img: Raw 2-D gray-scale image not yet reshaped
    """
    #Create a new dataframe to store all feature variables
    df = pd.DataFrame()
    
    #First feature = pixel values (0-255)
    #Unwrap 2D image to 1D using reshape
    img_reshape = img.reshape(-1)
    df['pixel'] = img_reshape
    
    #Next features = Gabor Filters features (various parameter combinations)
    num = 1 #To count numbers up in order to give Gabor filter features a label in the dataframe
    all_kernels = []
    ksize = 5
    gabor = {} #Create dictionary to know which # each gabor condition correspond to
    for theta in range(2): #Define number of thetas
        theta = theta/4. * np.pi
        for sigma in range(1,4): #Sigmas range 1 to 3
            for lamda in np.arange(0, np.pi, np.pi/4): #Range of wavelengths
                for gamma in (0.05, 0.5): #Gamma values range 0.05 or 0.5 (high aspect to medium aspect ratio kernel)

                    gabor_label = 'Gabor' +str(num) #Label Gabor feature columns with Gabor1, Gabor2, etc...

                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype = cv2.CV_32F)
                    all_kernels.append(kernel)

                    #Now apply filter to the image and add to feature column
                    f_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = f_img.reshape(-1)

                    df[gabor_label] = filtered_img

                    gabor[num] = ('theta',np.round(theta,3), 'sigma', np.round(sigma,3), 'lamda', np.round(lamda,3), 'gamma', np.round(gamma,3))
                    num += 1
                    
    #Next feature, Canny, Roberts, Sobel, Scharr, and Prewitt Edge Detection filters
    edges = cv2.Canny(img, 100, 200)
    edges_reshape = edges.reshape(-1)
    df['Canny_Edge'] = edges_reshape

    roberts_edge = roberts(img)
    roberts_reshape = roberts_edge.reshape(-1)
    df['Roberts_Edge'] = roberts_reshape

    sobel_edge = sobel(img)
    sobel_reshape = sobel_edge.reshape(-1)
    df['Sobel_Edge'] = sobel_reshape

    scharr_edge = scharr(img)
    scharr_reshape = scharr_edge.reshape(-1)
    df['Scharr_Edge'] = scharr_reshape

    prewitt_edge = prewitt(img)
    prewitt_reshape = prewitt_edge.reshape(-1)
    df['Prewitt_Edge'] = prewitt_reshape
    
  #Now add Gaussian Filter features
    for i in range (3,10,2):
        gaussian_img = nd.gaussian_filter(img, sigma=i)
        gaussian_reshape = gaussian_img.reshape(-1)
        df['Gaussian'+str(i)] = gaussian_reshape  
        
    #Now add Median Filter features
    for i in range (3,10,2):
        median_img = nd.median_filter(img, size=i)
        median_reshape = median_img.reshape(-1)
        df['Median'+str(i)] = median_reshape

    return df