# Code based on https://github.com/davidsandberg/facenet
# Code written by Pavel Jahoda

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet.src.facenet as facenet
import facenet.src.align.detect_face as detect_face
import cv2
import time
import math
import shutil
import dlib
import pickle

class IdentifyPeople:
    def __init__(self):
        self.image_size = 160
        self.minsize = 40
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.pnet, self.rnet, self.onet = None, None, None


    def ReadDetectAndEncode(self,imgPath,sess,n_jitters=0):
        img = misc.imread(imgPath, mode='RGB')
        bbs, landmarks = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        if len(bbs) != 1:
            return []

        img_list = [None]

        prewhitened = facenet.prewhiten(img)
        img_list[0] = prewhitened

        # Fixed normalization
        controlArray = np.expand_dims(np.zeros(1,dtype=np.int32),1)
        controlArray += np.expand_dims(np.ones(1,dtype=np.int32),1) * facenet.FIXED_STANDARDIZATION

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: img_list, self.phase_train_placeholder: False, self.control_placeholder:controlArray}
        img_encoding = sess.run(self.embeddings, feed_dict=feed_dict)

        if n_jitters:
            imgEncodings = img_encoding

            img = dlib.load_rgb_image(imgPath)
            augmented_images = dlib.jitter_image(img, num_jitters=n_jitters)

            for augmented_image in augmented_images:
                prewhitened = facenet.prewhiten(augmented_image)
                img_list[0] = prewhitened

                # Run forward pass to calculate embeddings
                feed_dict = {self.images_placeholder: img_list, self.phase_train_placeholder: False, self.control_placeholder:controlArray}
                img_encoding = sess.run(self.embeddings, feed_dict=feed_dict)

                imgEncodings = np.concatenate((imgEncodings,img_encoding),axis=0)

            return np.average(imgEncodings,axis=0)

        return img_encoding[0]

    # Bresenham's line algorithm to sample K out of N
    def Sample(self,N, K):
        result = [0 for i in range(N)]
        dx = N - 0
        dy = K - 0
        D = 2 * dy - dx
        y = 0

        if not D == D + 2 * dy:
            for x in range(N):
                if D > 0:
                    result[x] = 1
                    y = y + 1
                    D = D - 2 * dx
                D = D + 2 * dy

        return result

    def LastPhase(self,folderPath,sess):
        # delete folders with less than 10 images
        # check again if two folders do not contain same person, by sampling 10
        # images from each folder and averaging facial encodings
        identities = os.listdir(folderPath)

        folderNameAndEncodings = []

        for identityFolder in identities:
            imagesPaths = os.listdir(folderPath+'/'+identityFolder)
            sampledImages = []
            if len(imagesPaths) < 10:
                sampledImages = imagesPaths
            else:
                # sample 10 images
                sortedImages = []
                array = self.Sample(len(imagesPaths),10)
                for item in imagesPaths:
                    number = item.split('.')[0]
                    sortedImages.append(int(number))
                sortedImages.sort()
                for i in range(len(sortedImages)):
                    if array[i] == 1:
                        sampledImages.append(sortedImages[i])

                for i in range(len(sampledImages)):
                    sampledImages[i] = str(sampledImages[i])+'.png'

            img_encodings = []
            for sampledImage in sampledImages:
                img_encoding = self.ReadDetectAndEncode(folderPath+'/'+identityFolder+'/'+sampledImage,sess,n_jitters=4)
                img_encodings.append(img_encoding)
            img_encodings = np.array(img_encodings)

            # TODO possible improvement with matrices
            folderEncoding = np.average(img_encodings,axis=0)
            folderNameAndEncodings.append((identityFolder,folderEncoding))

        for i in range(len(folderNameAndEncodings)):
            distances = []
            for j in range(i+1,len(folderNameAndEncodings)):
                dist = facenet.distance([folderNameAndEncodings[i][1]], [folderNameAndEncodings[j][1]], distance_metric=0)
                distances.append(dist)

            if len(distances) != 0:
                minIndex = np.argmin(distances)
                if distances[minIndex] < 0.8:

                    # merge two folders
                    imagesToBeMoved = os.listdir(folderPath+'/'+folderNameAndEncodings[i][0])
                    for imageToBeMoved in imagesToBeMoved:
                        try:
                            os.rename(folderPath+'/'+folderNameAndEncodings[i][0]+'/'+imageToBeMoved,
                                      folderPath+'/'+folderNameAndEncodings[i+1+minIndex][0]+'/'+imageToBeMoved)
                            for face in self.ArrayOfTimeAndLocations[int(imageToBeMoved.split('.')[0])]:
                                if face != [] and str(face[0]) == folderNameAndEncodings[i][0]:
                                    face[0] = folderNameAndEncodings[i+1+minIndex][0]

                        except Exception as ex:
                            print(ex)

        # remove folders with less than 10 facial images
        for identityFolder in identities:
            imagesPaths = os.listdir(folderPath + '/' + identityFolder)
            if len(imagesPaths) < 10:
                shutil.rmtree(folderPath+'/'+identityFolder)



    def CreateFoldersByIdentity(self, folderPath, OutputFolderPath):

        with tf.Graph().as_default():

            with tf.Session() as sess:

                with open('Network/locations.p', 'rb') as f:
                    self.ArrayOfTimeAndLocations = pickle.load(f)

                if not os.path.exists(OutputFolderPath):
                    os.makedirs(OutputFolderPath)

                # Load MTCNN alignment model
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

                # Load the FaceNet model
                print('Loading FaceNet model')
                protobuf_file = 'Network/201804_model/20180402-114759.pb'
                facenet.load_model(protobuf_file)

                # Get input and output tensors
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')


                directoryWithPictures = os.listdir(folderPath)
                known_encodings = []

                maximum = 0
                for item in directoryWithPictures:
                    number = item.split('_')[0]
                    if maximum < int(number):
                        maximum = int(number)

                nOfFacesInEachFrame = np.zeros((maximum + 1), dtype=int)

                for item in directoryWithPictures:
                    number = item.split('_')[0]
                    nOfFacesInEachFrame[int(number)] += 1

                foundFirstFace = False
                print('Identify people in video')
                for frameNumber in range(len(nOfFacesInEachFrame)):
                    try:
                        if frameNumber % 100 == 0:
                            print('Frame number', frameNumber)
                        if nOfFacesInEachFrame[frameNumber] > 0:
                            if foundFirstFace:
                                distancesAndIndexes = []
                                for face_number in range(nOfFacesInEachFrame[frameNumber]):
                                    imgPath = folderPath + '/' + str(frameNumber) + '_' + str(face_number) + '.png'

                                    img_encoding = self.ReadDetectAndEncode(imgPath,sess)
                                    if len(img_encoding) == 0:
                                        continue

                                    if True:
                                        distances = []
                                        for known_encoding in known_encodings:
                                            dist = facenet.distance([img_encoding], [known_encoding], distance_metric=0)
                                            distances.append(dist)

                                        index = np.argmin(distances)
                                        distancesAndIndexes.append((distances[index], index, face_number))

                                distancesAndIndexes.sort()
                                cannotBeThem = []
                                for cnt, item in enumerate(distancesAndIndexes):
                                    imgPath = folderPath + '/' + str(frameNumber) + '_' + str(item[2]) + '.png'
                                    if item[1] not in cannotBeThem and item[0] < 0.8:
                                        os.rename(imgPath, OutputFolderPath+'/'+str(item[1]) + '/' + str(frameNumber) + '.png')
                                        for face in self.ArrayOfTimeAndLocations[frameNumber]:
                                            if face != []:
                                                face[0] = str(item[1])

                                        cannotBeThem.append(item[1])
                                    else:
                                        img_encoding = self.ReadDetectAndEncode(imgPath,sess, n_jitters=100)
                                        if len(img_encoding) == 0:
                                            continue

                                        # -------------------------------------------------------------------
                                        known_encodings.append(img_encoding)
                                        if not os.path.exists(OutputFolderPath+'/'+str(len(known_encodings) - 1)):
                                            os.makedirs(OutputFolderPath+'/'+str(len(known_encodings) - 1))

                                        os.rename(imgPath, OutputFolderPath+'/'+str(len(known_encodings) - 1) + '/' + str(frameNumber) + '.png')
                                        for face in self.ArrayOfTimeAndLocations[frameNumber]:
                                            if face != []:
                                                face[0] = str(len(known_encodings) - 1)

                            else:
                                foundFace = False
                                temporary = 0
                                for face_number in range(nOfFacesInEachFrame[frameNumber]):
                                    imgPath = folderPath + '/' + str(frameNumber) + '_' + str(face_number) + '.png'

                                    img_encoding = self.ReadDetectAndEncode(imgPath,sess, n_jitters=100)
                                    if len(img_encoding) == 0:
                                        temporary += 1
                                        continue

                                    known_encodings.append(img_encoding)

                                    if not os.path.exists(OutputFolderPath+'/'+str(face_number - temporary)):
                                        os.makedirs(OutputFolderPath+'/'+str(face_number - temporary))
                                    #
                                    foundFace = True

                                    os.rename(imgPath, OutputFolderPath + '/' + str(face_number - temporary) + '/' + str(frameNumber) + '.png')
                                    for face in self.ArrayOfTimeAndLocations[frameNumber]:
                                        if face != []:
                                            face[0] = str(face_number - temporary)

                                if foundFace:
                                    foundFirstFace = True

                    except Exception as ex:
                        print(ex)

                self.LastPhase('Identities',sess)
                if os.path.exists(folderPath):
                    shutil.rmtree(folderPath, ignore_errors=False, onerror=None)
                with open('Network/locations.p', 'wb') as f:
                    pickle.dump(self.ArrayOfTimeAndLocations, f)
