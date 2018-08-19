# Code written by Pavel Jahoda

import DecisionNetwork as DecisionNetwork
from VideoProcessing import VideoProcessing
from VideoDownload import DownloadVideo
from IdentifyPeople import IdentifyPeople
from DetectFaces import DetectFaces
import sys
import optparse

parser = optparse.OptionParser()
videoDownload = DownloadVideo()

parser.add_option('-k', '--keep',
    action="store", dest="identity",
    help="Keep folders with facial images divided by identity of the person", default="no")

parser.add_option('-o', '--only_recognize',
    action="store", dest="recognize",
    help="Only detect and recognize people in video", default="no")

options, args = parser.parse_args()

for url in args:
    videoDownload = DownloadVideo()
    try:
        videoDownload.Download(str(url))
    except Exception as exc:
        print('Wrong url')
        continue

    # try:
    detectFaces = DetectFaces()
    identifyPeople = IdentifyPeople()
    video = VideoProcessing()
    height, width, nOfFrames = detectFaces.DetectFacesFromVideo('input_video.mkv','DetectedFaces')
    identifyPeople.CreateFoldersByIdentity('DetectedFaces','Identities')
    if options.recognize == 'yes':
        continue

    if options.identity == 'no':
        video.CreateBatches('Identities')
    else:
        video.CreateBatches('Identities', True)
    rectangleList, rectanglesNoAmbiguity = DecisionNetwork.TestWholeNewFolder(False,True,False)

    fps = float(nOfFrames)/float(videoDownload.duration)

    video.ShowResult(rectangleList=rectangleList,width=width,height=height,rectanglesNoAmbiguity=rectanglesNoAmbiguity,fps=fps) #TODO
    videoDownload.MergeAudioAndVideo(video.outVideosDirectory+'/'+video.nameNotTaken,video.nameNotTaken)
    print('Done. Output video is ready.')
    # except Exception as exc:
    #     print('Something went wrong')
    #     continue
