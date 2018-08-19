# Code written by Pavel Jahoda

import math
import cv2
import dlib
import numpy as np
import os
import pickle

class DetectFaces:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.nOfDetectedPeople = 0
        self.constant = 20
        self.ArrayOfFaces = []


    def UseFaceSurroundings(self, factor, imageWidth, imageHeight,x,y,w,h):
        wPrevious = w
        hPrevious = h
        howMuch = factor

        tmp = (imageWidth - (x + w)) / w
        tmp2 = x / w
        tmp3 = y / h
        tmp4 = (imageHeight - (y + h)) / h

        minimum = min([howMuch-1, tmp, tmp2, tmp3, tmp4])
        minimum += 1

        w = int(math.floor(w * minimum))
        h = int(math.floor(h * minimum))
        x = x - int(math.floor(wPrevious * (((minimum - 1) / 2) + 1) - wPrevious))
        y = y - int(math.floor(hPrevious * (((minimum - 1) / 2) + 1) - hPrevious))

        return (x,y,w,h)


    def FindFaceResizeAndGrayScale(self, counter, frame, locations,dirToOutputFaces, useFaceSurroundings = True): #x,y,w,h

        img = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        zeros = False

        if len(locations) != 0 and counter % 30 != 0:
            faceLocations = []

            for location in locations:
                xPrev = max(location[0] - self.constant // 2, 0)
                yPrev = max(location[1] - self.constant // 2, 0)

                newGray = gray[yPrev:yPrev + location[3] + self.constant, xPrev:xPrev + location[2] + self.constant]
                dets = self.detector(newGray,0)

                if len(dets) == 1:
                    left = max(dets[0].left(),0)
                    top = max(dets[0].top(),0)
                    faceLocations.append((left,top,dets[0].right()-left,dets[0].bottom()-top))

        else:
            zeros = True
            faceLocations = []
            dets = self.detector(gray, 0)

            if len(dets) == 1:
                left = max(dets[0].left(), 0)
                top = max(dets[0].top(), 0)
                faceLocations.append((left, top, dets[0].right() - left, dets[0].bottom() - top))

        if len(faceLocations) == 0 or len(faceLocations) < len(locations):
            if not zeros:
                return self.FindFaceResizeAndGrayScale(counter,frame,[], dirToOutputFaces)
            else:
                return []

        if zeros == False:
            for cnt in range(min(len(faceLocations), len(locations))):
                faceLocations[cnt] = (faceLocations[cnt][0]+locations[cnt][0],faceLocations[cnt][1]+locations[cnt][1],faceLocations[cnt][2], faceLocations[cnt][3])

        self.ArrayOfFaces = []
        for location in faceLocations:
            self.ArrayOfFaces.append([str(-1), counter, location[1], (frame.shape[1] - location[0]), location[1] + location[3], frame.shape[1] - (location[0] + location[2])])

        if useFaceSurroundings:
            for cnt, faceLocation in enumerate(faceLocations):
                faceLocations[cnt] = self.UseFaceSurroundings(1.5,img.shape[1],img.shape[0],faceLocation[0],faceLocation[1],faceLocation[2],faceLocation[3])


        for cnt, faceLocation in enumerate(faceLocations):
            face_image = gray[faceLocation[1]:faceLocation[1] + faceLocation[3], faceLocation[0]:faceLocation[0] + faceLocation[2]]
            if face_image.shape[0]*face_image.shape[1] >= 160*160:
                face_image = cv2.resize(face_image,(160,160), interpolation=cv2.INTER_AREA)
            else:
                face_image = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(dirToOutputFaces + '/' + str(counter) + '_' + str(cnt) + '.png',face_image)

        if zeros == True:
            return faceLocations
        else:
            for cnt in range(min(len(faceLocations),len(locations))): #TODO what happens when more faces are detected than before
                faceLocations[cnt] = (faceLocations[cnt][0],faceLocations[cnt][1], max(faceLocations[cnt][2],locations[cnt][2]), max(faceLocations[cnt][3],locations[cnt][2]))

            return faceLocations

    def DetectFacesFromVideo(self, pathToVideo, dirToOutputFaces):
        try:
            if not os.path.exists(dirToOutputFaces):
                os.makedirs(dirToOutputFaces)

            locations = []
            counter = 0
            nOfSteps = 5 # After not detecting face in a frame, wait nOfSteps frames until another detection attempt
            stepsTillNextDetection = 0
            ArrayOfTimeAndLocations = []
            # ArrayOfFaces = []
            width, height = 0, 0

            video_capture = cv2.VideoCapture(pathToVideo)

            while video_capture.isOpened():
                # Grab a single frame of video
                ret, frame = video_capture.read()

                # Bail out when the video file ends
                if not ret:
                    break

                ArrayOfFaces = []

                height = frame.shape[0]
                width = frame.shape[1]

                if stepsTillNextDetection == 0:
                    locations = self.FindFaceResizeAndGrayScale(counter, frame, locations, dirToOutputFaces)
                    self.nOfDetectedPeople += len(locations)

                if len(locations) == 0 and stepsTillNextDetection == 0:
                    stepsTillNextDetection = nOfSteps

                if stepsTillNextDetection:
                    stepsTillNextDetection -= 1

                # x, y, w, h to -1, frameNumber, top, right, bottom, left
                # from dlib (0,0) coordinates are at top left corner, but for cv2 rectangle its top right corner
                # for location in locations:
                #     ArrayOfFaces.append([str(-1), counter, location[1], (frame.shape[1]- location[0]), location[1] + location[3], frame.shape[1]-(location[0] + location[2])])
                if len(locations) == 0:
                    ArrayOfTimeAndLocations.append([])
                else:
                    ArrayOfTimeAndLocations.append(self.ArrayOfFaces)
                # ArrayOfTimeAndLocations.append(ArrayOfFaces)

                counter += 1

                if counter % 100 == 0:
                    print('Detected', str(self.nOfDetectedPeople), 'faces from frame', str(counter - 99), 'to frame', counter)
                    self.nOfDetectedPeople = 0

            with open('Network/locations.p', 'wb') as f:
                pickle.dump(ArrayOfTimeAndLocations, f)

            return height, width, counter

        except Exception as ex:
            print(ex)
            return False

        # return True