import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
import pywt

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mode

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defect_data.hdF5'
images = h5py.File(file_path)['jute_defect_imgs'][:]
labels = h5py.File(file_path)['jute_defect_labels'][:]

def IMG(i):
    return resize_img(images[i])
def resize_img(img):
    dim = (500, 500)
    return cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
def show(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img'+str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def blur(img):
    ksize = (10,10)
    return cv2.blur(img,ksize)


# Gabor filter
def get_gabor_img(img):
    num_filters = 16
    ksize = 10  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    depth = -1
    newimage = np.zeros_like(img)
    for theta in np.arange(0,np.pi,np.pi/num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), 
                                sigma, theta, 
                                lambd, gamma, 
                                psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
        np.maximum(newimage, image_filter, newimage)
    return newimage

def get_contours(img):
    edge_img = cv2.Canny(img,50,100)
    closing = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=2)
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the image
    contour_image = cv2.drawContours(closing, contours[1:], -1, (0, 255, 0), 2)
    return contours, contour_image

def is_border_cont(cont):
    x, y, wid, hgt = cv2.boundingRect(cont)
    # Remove all clusters from a 5 px space from border
    if x<=5 or y<=5 or x+wid>=img.shape[1]-5 or y+hgt>=img.shape[0]-5:
        return True
    return False

def cluster_contours(contours):
    contour_vect = []
    for cont in contours:
        if is_border_cont(cont):
            continue
        cont = np.reshape(cont,(-1,2))
        vect = cont.shape[0], *tuple(np.median(cont,axis=0))
        contour_vect.append(vect)
    agglo = AgglomerativeClustering(n_clusters=8)
    clustered = agglo.fit_predict(contour_vect)
    return clustered

def crop_defect_space(img):
    gabor = get_gabor_img(img)
    contours, contour_img = get_contours(gabor)
    clustered = cluster_contours(contours)
    dense = mode(clustered)
    dense_idx = np.where(clustered==dense)[0]
    dense_contours = [np.vstack(contours[i]) for i in dense_idx]
    
    # dense_clust = cv2.drawContours(img, dense_contours, -1, (0, 255, 0), 2)
    # show([img,dense_clust])
    
    defect_space = np.vstack(dense_contours)
    poly = cv2.approxPolyDP(defect_space, 3, True)
    x, y, width, height = cv2.boundingRect(poly)
    return img[y:y+height,x:x+width]

def glcm(img,npx=32):
    converted_image = np.round(img / 255.0 * npx-1).astype(np.uint8)
    img = converted_image
    # Define the distance and angle offsets for co-occurrence
    d = 1
    theta = 0
    # Compute the co-occurrence matrix
    co_mat = np.zeros((npx, npx), dtype=np.uint32)
    for i in range(d, img.shape[0]):
        for j in range(d, img.shape[1]):
            i_index = img[i, j]
            j_index = img[i-d, j+theta]
            co_mat[i_index, j_index] += 1
    # Normalize and return the co-occurrence matrix
    co_mat = co_mat.astype(np.float64)
    co_mat /= np.sum(co_mat)
    return co_mat.flatten()


img = IMG(55)

# defect_crop = crop_defect_space(img)
# comx = glcm(defect_crop)
# print(comx)

comx_features = []
for img in images:
    defect_crop = crop_defect_space(img)
    features = glcm(defect_crop)
    comx_features.append(features)
    
comx_features = np.array(comx_features)

X_train, X_test, y_train, y_test = train_test_split(comx_features,labels,test_size=0.3,random_state=0)

svc = SVC(decision_function_shape='ovo', kernel='rbf', random_state = 0)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# center, radius = cv2.minEnclosingCircle(poly)
# dense_clust = cv2.drawContours(img, dense_contours, -1, (0, 255, 0), 2)
# color = 255
# cv2.rectangle(dense_clust, (int(x),int(y)), (int(x+width), int(y+height)), color, 2)
# cv2.circle(dense_clust, tuple(map(int,center)), int(radius), color, 2)