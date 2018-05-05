# Pavel Jahoda
# from .. import src
import DecisionNetwork as DecisionNetwork
from VideoProcessing import VideoProcessing
from VideoDownload import DownloadVideo
import sys
import pickle
import cv2


videoDownload = DownloadVideo()
for url in sys.argv:
    videoDownload = DownloadVideo()
    try:
        videoDownload.Download(str(url))
    except Exception as exc:
        print('Wrong url')
        continue

    try:
        video = VideoProcessing()
        dirName, height, width, fps = video.ProcessVideo(videoDownload.duration)
        print(fps)
        video.CreateBatches(dirName)
        rectangleList, rectanglesNoAmbiguity = DecisionNetwork.TestWholeNewFolder(False,True,False)

        video.ShowResult(rectangleList=rectangleList,width=width,height=height,rectanglesNoAmbiguity=rectanglesNoAmbiguity,fps=fps) #TODO
        videoDownload.MergeAudioAndVideo(video.outVideosDirectory+'/'+video.nameNotTaken,video.nameNotTaken)
        print('Done. Output video is ready.')
    except Exception as exc:
        print('Something went wrong')
        continue
