# Pavel Jahoda

import face_recognition
import cv2
import imageio as im
import os
from PIL import Image
import shutil
import pickle
import math
import numpy as np
import operator

class VideoProcessing:
    def __init__(self):
        self.ArrayOfTimeAndLocations = []
        self.remove = []

    def ResizeAndGrayScale(self,path):
        img = Image.open(path)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img.save(path)
        img = im.imread(path)
        img = img[:, :, 0]
        im.imwrite(path, img)

    def CreateBatches(self,directory):
        BagsDir = 'Bags'

        if os.path.exists(BagsDir):
            shutil.rmtree(BagsDir, ignore_errors=False, onerror=None)

        if not os.path.exists(BagsDir):
            os.makedirs(BagsDir)
        NOfBags = 0

        directories = os.listdir(directory)
        last = -1
        for dir in directories:
            images = os.listdir(os.path.join(directory, dir))
            if len(images) > 10:

                numbers = []
                for image in images:
                    tmp = image.split('.')
                    numbers.append(int(tmp[0]))
                numbers.sort()
                last = numbers[0]
                for number in numbers:
                    if number - last <= 5:
                        if not os.path.exists(BagsDir + '/' + str(NOfBags)):
                            os.makedirs(BagsDir + '/' + str(NOfBags))
                    else:
                        NOfBags += 1
                        os.makedirs(BagsDir + '/' + str(NOfBags))
                    os.rename(directory + '/' + dir + '/' + str(number) + '.png',
                              BagsDir + '/' + str(NOfBags) + '/' + str(number) + '.png')
                    self.ResizeAndGrayScale(BagsDir + '/' + str(NOfBags) + '/' + str(number) + '.png')
                    for face in self.ArrayOfTimeAndLocations[number]:
                        if face != [] and str(face[0]) == dir:
                            face[0] = str(NOfBags)

                    last = number

                NOfBags += 1



        bags = os.listdir(BagsDir)
        tmpNumbers = os.listdir(BagsDir)
        numbers = [int(item) for item in tmpNumbers]
        numbers.sort()
        highestNumberOfBag = numbers[-1]
        highestNumberOfBag += 1

        # split bags containing too much images into bags that have number of images close to
        # average bag size of training dataset
        for bag in bags:
            images = os.listdir(os.path.join(BagsDir, bag))
            if len(images) > 100:
                limit = (int(len(images) / 60) + 1) if ((len(images) / 60) - int(len(images) / 60)) > 0.5 else int(
                    len(images) / 60)
                limit = int(len(images) / limit)

                numbers = []
                for image in images:
                    tmp = image.split('.')
                    numbers.append(int(tmp[0]))
                numbers.sort()
                cnt = 0
                for number in numbers:
                    cnt += 1
                    if cnt >= limit:
                        if cnt % limit == 0:
                            highestNumberOfBag += 1
                        if not os.path.exists(BagsDir + '/' + str(highestNumberOfBag)):
                            os.makedirs(BagsDir + '/' + str(highestNumberOfBag))
                        os.rename(BagsDir + '/' + str(bag) + '/' + str(number) + '.png',
                                  BagsDir + '/' + str(highestNumberOfBag) + '/' + str(number) + '.png')

                        for face in self.ArrayOfTimeAndLocations[number]:
                            if face != [] and str(face[0]) == str(bag):
                                face[0] = str(highestNumberOfBag)

                highestNumberOfBag += 1

        with open('Network/locations.p', 'wb') as f:
            pickle.dump(self.ArrayOfTimeAndLocations, f)

        # delete almost empty bags
        bags = os.listdir(BagsDir)
        for bag in bags:
            images = os.listdir(os.path.join(BagsDir, bag))
            if len(images) < 10:
                shutil.rmtree(os.path.join(BagsDir, bag), ignore_errors=False, onerror=None)


    # does this bag contain same pictures of same person as on the picture being investigated
    def Decision(self, matches):
        if len(matches) == 0:
            return 0

        nOfTrue = 0
        for item in matches:
            if item == True:
                nOfTrue += 1
        return (math.log( (len(matches)+1) ,2) * float(nOfTrue/float(len(matches))) )


    def ProcessVideo(self,durationInSeconds):
        video_capture = cv2.VideoCapture("input_video.mkv")
        VideoPicturesInBags = 'FacialImagesFromVideo'

        #nOfFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        #fps = video_capture.get(cv2.CAP_PROP_FPS)
        #print('Video has',fps,'frames per second.')

        if os.path.exists(VideoPicturesInBags):
            shutil.rmtree(VideoPicturesInBags, ignore_errors=False, onerror=None)

        frames = []
        temporary_frames = []
        frame_count = 0

        known_face_names = []
        known_face_encodings = []
        numberOfPicturesForEachPerson = []
        toThePowerOfWhat = []

        numberOfPeople = 0

        # This code finds all faces in a list of images using the CNN model.

        height = None;width = None
        # Open video file
        framesThatWeMightUse = []
        counter = 0
        previouslyContainedFace = False

        while video_capture.isOpened():
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Bail out when the video file ends
            if not ret:
                break

            counter += 1

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            frame = frame[:, :, ::-1]
            frames.append(frame)
            if counter % 2 == 0:
                height = frame.shape[0]
                width = frame.shape[1]

                temporary_frames.append(frame)
            else:
                framesThatWeMightUse.append(frame)

            frame_count += 1

            # Every 6 frames, batch process the list of frames to find faces
            if len(frames) == 6:

                temporary_batch_of_face_locations = face_recognition.batch_face_locations(temporary_frames, number_of_times_to_upsample=0)
                batch_of_face_locations = []

                # process only every seconds frame if number of faces found did not change between the every second frame
                for frame_number_in_batch, face_locations in enumerate(temporary_batch_of_face_locations):
                    if (len(face_locations) == 1 and previouslyContainedFace == False ) or (len(face_locations) == 0 and previouslyContainedFace == True):
                        previouslyContainedFace = not previouslyContainedFace
                        temporaryFrame = framesThatWeMightUse[frame_number_in_batch]
                        temporary_batch = face_recognition.batch_face_locations([temporaryFrame], number_of_times_to_upsample=0)

                        batch_of_face_locations.append(temporary_batch[0])
                        batch_of_face_locations.append(temporary_batch_of_face_locations[frame_number_in_batch])
                    else:
                        for i in range(2):
                            batch_of_face_locations.append(temporary_batch_of_face_locations[frame_number_in_batch])

                # Now let's list all the faces we found in all 6 frames
                for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                    number_of_faces_in_frame = len(face_locations)

                    face_encodings = face_recognition.face_encodings(frames[frame_number_in_batch], face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        topScore = 0
                        indexOfTopScore = 0
                        found = False
                        for index in range(len(known_face_encodings)):
                            matches = face_recognition.compare_faces(known_face_encodings[index], face_encoding, tolerance=0.60)

                            tmp = self.Decision(matches)
                            if tmp>topScore:
                                topScore = tmp
                                indexOfTopScore = index

                        if topScore > 0.5:
                            index = indexOfTopScore
                            name = known_face_names[index]
                            found = True
                            numberOfPicturesForEachPerson[index] += 1
                            if numberOfPicturesForEachPerson[index] == int(pow(3, toThePowerOfWhat[index])):
                                toThePowerOfWhat[index] += 1
                                known_face_encodings[index].append(face_encoding)


                        if not found:
                            name = str(numberOfPeople)
                            known_face_encodings.append([face_encoding])
                            known_face_names.append(name)
                            numberOfPicturesForEachPerson.append(1)
                            toThePowerOfWhat.append(1)
                            numberOfPeople += 1
                        face_names.append(name)

                    frame_number = frame_count - 6 + frame_number_in_batch
                    print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))
                    cnt = 0
                    temporaryList = []
                    ArrayOfFaces = []
                    for face_location in face_locations:
                        if not os.path.exists(VideoPicturesInBags + '/' + face_names[cnt]):
                            os.makedirs(VideoPicturesInBags + '/' + face_names[cnt])

                        top, right, bottom, left = face_location
                        face_image = frames[frame_number_in_batch][top:bottom, left:right]
                        im.imwrite(VideoPicturesInBags + '/' + face_names[cnt] + '/' + str(frame_number) + '.png',face_image)
                        temporaryList = [face_names[cnt], frame_number, top, right, bottom, left]
                        ArrayOfFaces.append(temporaryList)

                        cnt += 1

                    # if temporaryList == []:
                    #     ArrayOfFaces.append(temporaryList)

                    self.ArrayOfTimeAndLocations.append(ArrayOfFaces)

                    face_names = []

                # Clear the frames array to start the next batch
                frames = []
                framesThatWeMightUse = []
                temporary_frames = []


        video_capture.release()

        fps = float(frame_count) / float(durationInSeconds)

        return VideoPicturesInBags, height, width, fps

    def FindUnusedName(self,dir,name):

        nameNotTaken = ''
        counter = 0
        while True:
            if not os.path.exists(dir+'/'+name+str(counter)+'.mp4'):
                nameNotTaken = name+str(counter)+'.mp4'
                break
            counter += 1

        return nameNotTaken

    def ConvertToHoursMinutesSecondsAndCentiSeconds(self,seconds):
        hours = 0
        minutes = 0
        secondsReturn = 0
        centiseconds = 0
        rest = 0
        if seconds >= 3600:
            rest = seconds%3600
            hours = int(seconds/3600)
            seconds = rest
        if seconds >= 60:
            rest = seconds%60
            minutes = int(seconds/60)
            seconds = rest
        if seconds % 1 != 0:
            centiseconds = int((seconds % 1)*100)
            secondsReturn = int(seconds)
        else:
            secondsReturn = int(seconds)

        returnString = str(hours)+':'+str(minutes)+':'+str(secondsReturn)+':'+str(centiseconds)
        return returnString

    # write output video into OutputVideos and csv file into OutputInformation directory
    def ShowResult(self, rectangleList,width,height,rectanglesNoAmbiguity,fps):
        print('Creating Video')
        video_capture = cv2.VideoCapture("input_video.mkv")
        frame_count = 0

        self.outVideosDirectory = 'OutputVideos'
        self.outInformationDirectory = 'OutputInformation'
        if not os.path.exists(self.outVideosDirectory):
            os.makedirs(self.outVideosDirectory)

        if not os.path.exists(self.outInformationDirectory):
            os.makedirs(self.outInformationDirectory)

        self.nameNotTaken = self.FindUnusedName(self.outVideosDirectory,'output_video')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video = cv2.VideoWriter(os.path.join(self.outVideosDirectory,self.nameNotTaken), fourcc, fps, (width, height))

        while video_capture.isOpened():
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                print('frame:',frame_count)
            try:
                for rectangle in rectangleList: #TODO OPTIMIZE
                    if rectangle != [] and rectangle[0] == frame_count:
                        top = rectangle[1]; right = rectangle[2]; bottom = rectangle[3]; left = rectangle[4]

                        # Draw a box around the face with decision ambiguity
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 4)

            except Exception as ex:
                pass

            try:
                for rectangle in rectanglesNoAmbiguity: #TODO OPTIMIZE
                    if rectangle != [] and rectangle[0] == frame_count:
                        top = rectangle[1]; right = rectangle[2]; bottom = rectangle[3]; left = rectangle[4]

                        # Draw a box around the face with no decision ambiguity
                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (105, 105, 105), 4)

            except Exception as ex:
                pass

            try:
                # Create a csv file containing sections with decision ambiguity
                # TODO optimize
                dataForCSVFile = [] #folder, minTime, maxTime
                for item in rectangleList:
                    found = False
                    for dataItem in dataForCSVFile:
                        if dataItem[0] == item[-1]:
                            found = True
                            if int(item[0]) < int(dataItem[1]):
                                dataItem[1] = item[0]
                            if int(item[0]) > int(dataItem[2]):
                                dataItem[2] = item[0]
                    if not found:
                        dataForCSVFile.append([item[-1],item[0],item[0]])
                dataForCSVFile = sorted(dataForCSVFile, key=operator.itemgetter(1))

                with open(self.outInformationDirectory+'/'+self.nameNotTaken+'.txt','w') as file:
                    for dataItem in dataForCSVFile:
                        for counter, data in enumerate(dataItem):
                            if counter != 0:
                                if counter != (len(dataItem)-1):
                                    file.write(self.ConvertToHoursMinutesSecondsAndCentiSeconds(float(data/float(fps))) + ', ')
                                else:
                                    file.write(self.ConvertToHoursMinutesSecondsAndCentiSeconds(float(data / float(fps))))
                        file.write('\n')

                pass
            except Exception as ex:
                print('something went wrong with creating csv file')
                pass

            video.write(frame)

            frame_count += 1

        video.release()
        video_capture.release()
        cv2.destroyAllWindows()
        print('Video created')

