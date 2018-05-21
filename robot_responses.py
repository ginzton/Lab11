#!/usr/bin/env python3

import asyncio
import sys

import numpy as np
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps


import cozmo
import re
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog
from skimage.color import rgb2gray
import cv2

def cozmo_turn_in_place(robot, angle, speed):
	"""Rotates the robot in place.
		Arguments:
		robot -- the Cozmo robot instance passed to the function
		angle -- Desired distance of the movement in degrees
		speed -- Desired speed of the movement in degrees per second
	"""
	robot.turn_in_place(degrees(angle), speed=degrees(speed)).wait_for_completed()


def cozmo_drive_straight(robot, dist, speed):
    """Drives the robot straight.
        Arguments:
        robot -- the Cozmo robot instance passed to the function
        dist -- Desired distance of the movement in millimeters
        speed -- Desired speed of the movement in millimeters per second
    """

    speed_instance = speed_mmps(speed)
    robot.drive_straight(distance_mm(dist), speed_mmps(speed)).wait_for_completed()

class ImageClassifier:

    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir + "*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)
        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return (data, labels)

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
        return (feature_data)

    def extract_image_feature(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################


        gray_img = rgb2gray(data)
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

        feature_data = fd

        #        print("feature_data dimension", feature_data.shape)
        # Please do not modify the return type below
        return (feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        ########################
        ######## YOUR CODE HERE
        ########################

        self.classifer = SVC(kernel = 'linear', C=1)
        self.classifer.fit(train_data, train_labels)

        #        self.classifer = LinearSVC(kernel='linear', C=1).fit(train_data, train_labels)

 #       pass

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


# Define a decorator as a subclass of Annotator; displays battery voltage
# class BatteryAnnotator(cozmo.annotate.Annotator):
#     def apply(self, image, scale):
#         d = ImageDraw.Draw(image)
#         bounds = (0, 0, image.width, image.height)
#         batt = self.world.robot.battery_voltage
#         text = cozmo.annotate.ImageText('BATT %.1fv' % batt, color='green')
#         text.render(d, bounds)


def run(robot: cozmo.robot.Robot):
    '''The run method runs once the Cozmo SDK is connected.'''

    try:


       img_clf = ImageClassifier()
       (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
       train_data = img_clf.extract_image_features(train_raw)
       img_clf.train_classifier(train_data, train_labels)
       print("finished the training")
       print("let's start imaging!")

       i = 0
       n_truck = 0
       n_plane = 0
       n_order = 0

       while True:

           robot.set_lift_height(0, in_parallel=False, duration=1).wait_for_completed()
           robot.set_head_angle(degrees(5), in_parallel=False, duration = 1).wait_for_completed()
           image = []

           event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

           i += 1

           print("i",i)

            #convert camera image to opencv format

 #           cv2.imwrite("picture_" + str(i) + ".bmp",np.asarray(event.image))
           original_image = event.image

           blur = cv2.GaussianBlur(np.asarray(original_image), (5,5),0)
           image.append(blur)
           print("image dimension",len(image))

           extracted_features = img_clf.extract_image_features(image)
           print("extracted_features",extracted_features)
           predicted_labels = img_clf.predict_labels(extracted_features)
           print(predicted_labels)

           if predicted_labels == 'truck':

               n_truck += 1

               if n_truck > 10:
                   robot.say_text("truck",in_parallel=False,num_retries=5).wait_for_completed()
                   robot.set_lift_height(180, in_parallel=False, duration=0.5).wait_for_completed()
                   robot.set_lift_height(0, in_parallel=False, duration = 0.5).wait_for_completed()

                   n_truck = 0

           elif predicted_labels == 'plane':

               n_plane += 1

               if n_plane > 10:
                   robot.say_text("plane",in_parallel=False,num_retries=5).wait_for_completed()
                   cozmo_turn_in_place(robot, 90, 90)
                   cozmo_turn_in_place(robot, -90, 90)


                   n_plane = 0

           elif predicted_labels == 'order':

               n_order += 1

               if n_order > 10:
                   robot.say_text("order", in_parallel=False, num_retries=5).wait_for_completed()
                   cozmo_drive_straight(robot, 50, 50)
                   cozmo_drive_straight(robot, -50, 50)

                   n_order = 0


if __name__ == '__main__':

    cozmo.camera.Camera.image_stream_enabled = True
    cozmo.run_program(run)#, use_viewer = True, force_viewer_on_top = True)

