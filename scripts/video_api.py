import os
from moviepy.editor import *
os.environ["IMAGEMAGICK_BINARY"] = "./imagemagick.exe"

import librosa
duration = librosa.get_duration(path='audio/hello.mp3')

def resize_func(t):
    if t < 2: return 1 + 0.2*t  # Zoom-in.
    elif 2 <= t <= 4: return 1 + 0.2*2  # Stay.
    else: return 1 + 0.2*(duration-t)  # Zoom-out.

def position_func(t):
    if t < 2: 
        print("I love it")
        return (-20*t, -20*t)  # Zoom-in.
    elif 2 <= t <= 4: return (-20*2, -20*2)  # Stay.
    else: return (-20*2 + 20*(t-4), -20*2 + 20*(t-4))  # Zoom-out.

image = ImageClip('img/test2.jpg', duration=duration)
image = image.resize(resize_func)
image = image.set_position(position_func)
image = image.fadein(1)
image = image.fadeout(1)
image.fps = 60
compvideo = CompositeVideoClip([image], size=image.size)
audioclip = AudioFileClip("audio/hello.mp3")
compaudio = CompositeAudioClip([audioclip])
compvideo.audio = compaudio
compvideo.write_videofile('video/test.mp4')


# videoclip = VideoFileClip("filename.mp4")
# audioclip = AudioFileClip("audioname.mp3")
# new_audioclip = CompositeAudioClip([audioclip])
# videoclip.audio = new_audioclip
# videoclip.write_videofile("new_filename.mp4")
