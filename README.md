## Deep learning AI, that recognizes when are people uncertain.  [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)


### How to Install:

After you have installed python3 (3.6), virtualenv (sudo apt install virtualenv) and ffmpeg, clone this repository and in the cloned directory DecisionAmbiguityRecognition run these commands:

```
virtualenv -p python3 env
source env/bin/activate
pip3 install -r requirements.txt
```

### How to use:
Activate virtual environment( source env/bin/activate ) and the run:

```
python3 src/main.py url
```


Where url is youtube video URL or several youtube video URL's separeted by space

Once the program finished detecting decision ambiguity from facial images, you will have video in folder OutputVideos and csv file in OutputInformation.

In the output video, decision ambiguity is marked by red rectangle around person's face. Gray rectangle around person's face occurs, when our AI determined that person is not ambiguous. All the other instances when facial images aren't surrounded by rectangle, are simply not used by our AI either because face_recognition did not recognize facial images, or the section with facial image was too short.

In the output csv file, you will find sections of decision ambiguity.
Format: x, y

x = hours:minutes:seconds:centiseconds representing start of section with decision ambiguity<br/>
y = hours:minutes:seconds:centiseconds representing end of section with decision ambiguity 


### Requirements

- ffmpeg
- python3.6
- virtualenv
- cmake


### Links
[Link to video demonstration](https://youtu.be/LNgvCIBq1b4) <br/>
[Link to publication](bit.ly/2rotkUJ) <br/>
[Link to dataset](http://cmp.felk.cvut.cz/~jahodpa1/millionaire/)

### Citation
Please cite us:
```
@InProceedings{ambiguity2018,
  author =       {Pavel Jahoda and Antonin Vobecky and Jan Cech and Jiri Matas},
  title =        {Detecting Decision Ambiguity from Facial Images},
  booktitle = {IEEE International Conference on Automatic Face and Gesture Recognition},
  year =      {2018}
}
```
