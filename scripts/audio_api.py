from gtts import gTTS
tts = gTTS('Hello.  My name is Content Farm Zero and this is an example of audio output.')
tts.save('audio/hello.mp3')

import librosa
duration = librosa.get_duration(path='audio/hello.mp3')
print(duration)
