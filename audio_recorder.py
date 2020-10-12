import os
import threading
from datetime import datetime
import ffmpeg
import time
import speech_recognition as sr
import jieba
import jieba.posseg as pseg
import magic
import json

class MicCapture:
    """[Receiving microphone streaming audio, using multi-threading method to reduce the problem of buffer stacking audio.]
    """    
    def __init__(self, cam_ip, audio_spath, audio_stime):
        """[Initial MicCapture class.]

        Args:
            cam_ip ([str]): [Ip camera rtsp url.]
            audio_spath ([str]): [Path of record path.]
            audio_stime ([int]): [How long times need to record for audio.]
        """        
        ## Init setting
        self.isstop = False
        self.cam_ip = cam_ip
        self.audio_spath = audio_spath
        self.audio_stime = audio_stime
        self.pre_file_new = str()

        if not(os.path.exists(self.audio_spath)):
            os.mkdir(self.audio_spath)

    def start(self):
        """[Start capture audio from microphone of ip cam.]
        """        
        print("mic started!")
        threading.Thread(target=self.queryaudio, daemon=True, args=()).start()

    def stop(self):
        """[Switch to stop infinite loop.]
        """        
        self.isstop = True
        print("mic stopped!")
   
    def getaudio(self):
        """[When there is a need for audio, the latest audio is returned.(For Quadro W analyze)]
        """        

        ## List all files in the directory
        audio_lists = os.listdir(self.audio_spath)

        ## if no audio file
        if len(audio_lists) == 0:
            file_new = None
        ## If audio file only have one.
        elif len(audio_lists) == 1:
            file_new = os.path.join(self.audio_spath, audio_lists[0])
            ## Check wheather it has been analyzed.
            if file_new == self.pre_file_new:
                file_new = None
            else:
                self.pre_file_new = file_new
        else:
            ## Sort by create time & get latest one.
            audio_lists.sort(key=lambda fn:os.path.getmtime(self.audio_spath + "/" + fn))
            file_new = os.path.join(self.audio_spath, audio_lists[-1])

            if len(audio_lists) > 5: # If havemore than "5" audio files, delete them.
                ## Delete outdata audio data without latest one.
                for i in range(len(audio_lists)-1):
                    try:
                        os.remove(os.path.join(self.audio_spath, audio_lists[i]))
                    except OSError as e:
                        pass
            self.pre_file_new = file_new
        
        return(file_new)

        
    def queryaudio(self):
        """[Collect audio.]
        """        
        while (not self.isstop):
            ## Define audio save name
            now = datetime.now()
            audio_name = now.strftime("%H_%M_%S") + ".wav"
            spath = os.path.join(self.audio_spath, audio_name)

            ## Capture audio from ip cam microphone
            f = ffmpeg.input(self.cam_ip, allowed_media_types="audio",  rtsp_transport="tcp", t=self.audio_stime)["a"]
            f = f.filter("volume", 1).output(spath, acodec="pcm_s16le", ac=2, ar="16k")
            f = f.overwrite_output()
            f = f.global_args("-loglevel", "error")
            f = f.run(capture_stdout=True)

class AudioRecorder:
    """[Audio class based on librosa, Speach-to-text and Jieba.]
    """    
    def __init__(self, cam_ip, audio_spath, audio_stime, pos_path, dict_path):
        """[Iniial AudioRecorder class.]

        Args:
            cam_ip ([str]): [Ip camera rtsp url.]
            audio_spath ([str]): [Path of record.]
            audio_stime ([int]): [How long times need to record for audio.]
            pos_path ([str]): [Path of focused POS ]
            dict_spath ([str]): [Path of added dictionary.]
        """        
        ## Init setting
        self.open = True
        ## Add new dictionary
        jieba.load_userdict(dict_path)
        ## Define what POS(part-of-speech) care
        with open(pos_path) as json_file: 
            self.pos_dict = json.load(json_file) 

        ## Init speak result
        self.speak = "No speaking"
        self.speaking_word = [ ["" for j in range(5)] for i in range(12)]

        ## Microphone connection.
        self.audio_cap = MicCapture(cam_ip, audio_spath, audio_stime)
        ## Microphone Capture thread start
        self.audio_cap.start()
        ## Pause for to ensure the audio has been filled
        time.sleep(audio_stime)

    # Audio starts being recorded
    def record(self):
        while (self.open == True):
            ## Get the latest audio record.
            self.audio = self.audio_cap.getaudio()

            ## Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.
            ## Thread for speech recontition.
            speech_thread = threading.Thread(target=self.speach2text, daemon=True, args=())

            ## Start thread of speech recontition.
            speech_thread.start()
            
            ## Wait thread of speech recontition.
            speech_thread.join()

    def stop(self):
        """[Finishes the audio recording therefore the thread too]
        """        
        if self.open == True:
            self.open = False
    
    def start(self):
        """[Launches the audio recording function using a thread]
        """        
        audio_thread_cam1 = threading.Thread(target=self.record, daemon=True)
        audio_thread_cam1.start()

    def speach2text(self):
        """[Transform speach to text and split it by POS(part-of-speech)]
        """        
        if self.audio is not None :
            if (magic.from_file(self.audio) != "empty"):
                ## Speach Recpgnize by google Speach-to-text
                r = sr.Recognizer()
                hellow = sr.AudioFile(self.audio)
                with hellow as source:
                    audio = r.record(source)
                try:
                    self.speak = r.recognize_google(audio, language="zh-TW")
                    
                    ## Cut the text with POS((part-of-speech) by jieba)
                    words = pseg.cut(self.speak)
                    self.speaking_word = [ [] for _ in range(12)]
                    for word in words:
                        if word.flag in self.pos_dict:
                            self.speaking_word[self.pos_dict[word.flag]].append(word.word)
                
                ## Audio not have speak
                except Exception as e:
                    self.speaking_word = [ [] for _ in range(12)]
                    self.speak = "No speaking"

                ## Fill list for show on QT table
                for i in range(len(self.speaking_word)):
                    for j in range(5 - len(self.speaking_word[i])):
                        self.speaking_word[i].append("")
            else:
                pass
        else:
            pass