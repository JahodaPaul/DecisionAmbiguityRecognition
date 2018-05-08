#Pavel Jahoda

from __future__ import unicode_literals
import youtube_dl
import os
import subprocess
import requests

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

class DownloadVideo:
    def __init__(self):
        self.DownloadNetwork()
        self.duration = 0
        self.videoTitle = 'input_video.mkv'
        self.audioTitle = 'audio.m4a'
        self.ydl_opts = {
            'outtmpl': self.videoTitle,
            'merge_output_format': 'mkv',
            'logger': MyLogger(),
            'progress_hooks': [self.my_hook],
        }

        self.audioOptions = {
            'outtmpl': self.audioTitle,
            'format': 'bestaudio/best',
            'audioformat': 'm4a',
            'logger': MyLogger(),
            'progress_hooks': [self.my_hook],
        }

    def DownloadNetwork(self):
        url = 'https://www.dropbox.com/s/mflziyzf208gskp/model_1out_BN_DO-07_TF_1-improvement-05-0.68.hdf5?dl=1'
        networkPath = './Network/model_1out_BN_DO-07_TF_1-improvement-05-0.68.hdf5'

        if not os.path.exists(networkPath):
            print('Downloading neural network for uncertainty detection')
            r = requests.get(url)

            with open(networkPath, 'wb') as f:
                f.write(r.content)

            # Retrieve HTTP meta-data
            print(r.headers['content-type'])
            print('Done downloading neural net')

    def my_hook(self,d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    def Download(self,url,videoName=''):
        print('Downloading')
        if os.path.exists(self.videoTitle):
            os.remove(self.videoTitle)

        if os.path.exists(self.audioTitle):
            os.remove(self.audioTitle)

        if videoName != '':
            if os.path.exists(videoName):
                os.remove(videoName)
            self.ydl_opts['outtmpl'] = videoName
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            self.duration = info['duration']

        with youtube_dl.YoutubeDL(self.audioOptions) as ydlAudio:
            ydlAudio.download([url])

        print('Done converting')

    def MergeAudioAndVideo(self,videoPath,videoName):
        print('Merging audio and video')
        pathToOutput = videoPath.split('/')[0]
        outputVideoName = videoName.split('.')[0]
        finalOutput = pathToOutput+'/'+outputVideoName
        commandToCall = 'ffmpeg -loglevel panic -i '+videoPath+' -i '+self.audioTitle+' -c copy '+finalOutput+'.mp4'
        processOne = subprocess.call(commandToCall.split(' '))
        if os.path.exists(videoPath):
            os.remove(videoPath)
        if os.path.exists(self.audioTitle):
            os.remove(self.audioTitle)
        print('Done Merging')
