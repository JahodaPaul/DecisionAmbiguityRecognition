# Code written by Pavel Jahoda

import cv2
import os
from PIL import Image
import shutil
import pickle
import math
import operator

class VideoProcessing:
    def __init__(self):
        self.ArrayOfTimeAndLocations = []
        self.remove = []

    def Resize(self,path):
        img = Image.open(path)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img.save(path)

    def CreateBatches(self,directory, keep=False):
        with open('Network/locations.p', 'rb') as f:
            self.ArrayOfTimeAndLocations = pickle.load(f)


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
                    if keep==False:
                        os.rename(directory + '/' + dir + '/' + str(number) + '.png',BagsDir + '/' + str(NOfBags) + '/' + str(number) + '.png')
                    else:
                        shutil.copyfile(directory + '/' + dir + '/' + str(number) + '.png',BagsDir + '/' + str(NOfBags) + '/' + str(number) + '.png')
                    self.Resize(BagsDir + '/' + str(NOfBags) + '/' + str(number) + '.png')
                    for face in self.ArrayOfTimeAndLocations[number]:
                        if face != [] and str(face[0]) == dir:
                            face[0] = str(NOfBags)

                    last = number

                NOfBags += 1

        if keep == False:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=False, onerror=None)

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


    def FindUnusedName(self,dir,name):

        nameNotTaken = ''
        counter = 0
        while True:
            if not os.path.exists(dir+'/'+name+str(counter)+'.mkv'):
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

            if frame_count % 100 == 0:
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

