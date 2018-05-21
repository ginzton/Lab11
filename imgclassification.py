#!/usr/bin/env python

##############
#### Your name:
##############

import numpy as np
import re
from sklearn import svm, metrics
from sklearn.svm import SVC
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

class ImageClassifier:
    
    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################

        feature_data = []

        for img in data:
            gray_img = rgb2gray(img)
            fd = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                  transform_sqrt=True)

#            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            # ax1.axis('off')
            # ax1.imshow(img, cmap=plt.cm.gray)
            # ax1.set_title('Input image')
            #
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255))
            #
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            # ax2.axis('off')
            # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            # ax2.set_title('Histogram of Oriented Gradients')
            # plt.show()

            feature_data.append(fd)

#        print("feature_data dimension", feature_data.shape)
        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        
        ########################
        ######## YOUR CODE HERE
        ########################

        self.classifer = SVC(kernel = 'linear', C=1)
        self.classifer.fit(train_data, train_labels)

        pass

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        ########################

        predicted_labels = self.classifer.predict(data)
        
        # Please do not modify the return type below
        return predicted_labels

      
def main():

    img_clf = ImageClassifier()

    # load images
#    (train_raw, train_labels) = img_clf.load_data_from_folder('./temp/')

    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    # #
    # # # convert images into features
    #temp_data = img_clf.extract_image_features(train_raw)
    #print("extracted data",temp_data)
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    #
    # # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("predicted labels", predicted_labels)
    print("test labels", test_labels)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
