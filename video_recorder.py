import sys
import cv2
import threading
import numpy as np
from datetime import datetime
import time as _time
import os
import csv
import pandas as pd

from preprocess import DLImgResize, LoadCategory, ImgResize, Pro2Score, LoadLabelMap
from load_model import DNNModel_Object, DNNModel_Pb
from load_cdcl import CDCLModel_Pb, Detect

from face_recognize.segmentation_CDCL.restnet101 import get_testing_model_resnet101

sys.setrecursionlimit(19390)

class IpcamCapture:
    """[Receiving camera streaming images, using multi-threading method to reduce the problem of buffer stacking frames.]
    """    
    def __init__(self, url, fps, analysis_sec):
        """[Initial IpcamCapture class]

        Args:
            url ([str]): [Ip camera rtsp url.]
            fps ([int]): [Ip camera fps.]
            analysis_sec ([int]): [Quadro W analysis interval.]
        """        
        ## Init setting
        self.frames = list()
        self.status = False
        self.isstop = False

        ## Set camera information
        self.fps = fps
        self.analysis_sec = analysis_sec ## Quadro W analyze interval

        ## Camera connection.
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps) ## Set fixed FPS

    def start(self):
        """[Start capture fram from ip cam.]
        """        
        ## Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.
        print("ipcam started!")
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        """[Switch to stop infinite loop.]
        """        
        self.isstop = True
        print("ipcam stopped!")
   
    def getframe(self):
        """[When there is a need for a frame, the latest frame is returned.(To show on GUI)]

        Returns:
            [numpy array]: [Current frame captured from query.]
        """        
        return self.frame

    def getframes(self):
        """[When there is a need for frmaes, the latest frames is returned.(For Quadro W analyze)]

        Returns:
            [list]: [List of multiple frames from query.]
        """        
        return self.frames
        
    def queryframe(self):
        """[Collect frame in query to avoid blocking.]
        """        
        frames = list()
        while (not self.isstop):
            ## get frame from cmaera
            self.status, self.frame = self.capture.read()
            
            ## Collect frames in the same Quadro W analyze interval
            if self.capture.get(cv2.CAP_PROP_POS_FRAMES)%(self.fps*self.analysis_sec) == 0.0:
                frames.append(self.frame)
                self.frames = frames
                frames = list()
            else:
                frames.append(self.frame)
        
        ## Release video capture
        self.capture.release()


class VideoRecorder():
    """[Video class based on openCV and Quardo W model.]
    """    
    def __init__(self, cam_ip, record_path, qtable_path, compress_result, fps, analysis_sec, prefix_name, audio_thread,
                 human_model_path, face_model_path, place_model_path, object_model_path, 
                 face_category_path, place_category_path, object_category_path):
        """[Iniial VideoRecorder class.]

        Args:
            cam_ip ([str]): [Ip camera rtsp url.]
            record_path ([str]): [Saved path of quadro W inforamtion.]
            qtable_path ([str]): [Saved path of Q table.]
            compress_result ([bool]): [Wheather to compress same quadro W inforamtion.]
            fps ([int]): [Ip camera fps.]
            analysis_sec ([int]): [Quadro W analysis interval.]
            prefix_name ([str]): [Prefix name before tensor node name.]
            audio_thread ([AudioRecorder class]): [Audio analysys stream thread.]
            human_model_path ([str]): [Path of detect human model.]
            face_model_path ([str]): [Path of recognition human model.]
            place_model_path ([str]): [Path of recognition place model.]
            object_model_path ([str]): [Path of recognition object model.]
            face_category_path ([str]): [Path of recognition human category.]
            place_category_path ([str]): [Path of recognition place category.]
            object_category_path ([str]): [Path of recognition object category.]
        """                 
        ## Init setting
        self.open = True

        self.record_path = record_path
        self.qtable_path = qtable_path
        self.compress_result = compress_result
        self.cam = prefix_name ## Use prefix_name to represent camera

        self.pre_result = [None for _ in range(23)]
        self.result = ["None" for _ in range(23)]
        self.speaking_word = [ ["","","","",""] for _ in range(12)]

        ## Camera connection.
        self.video_cap = IpcamCapture(url=cam_ip, fps=fps, analysis_sec=analysis_sec)
        ## Ipcam Capture thread start
        self.video_cap.start()
        ## Pause for 1 second to ensure the image has been filled
        _time.sleep(1)

        ## Set each model input size
        self.place_size = (224, 224)
        self.human_size = (444, 250)
        self.face_size = (224, 224)

        ## Set inforamtion of segment human region(human_model)
        gpus = 1
        scale = [1]
        input_pad = 8

        ## Get each model category
        self.face_category = LoadCategory(category_path=face_category_path)
        self.place_category = LoadCategory(category_path=place_category_path)
        self.object_category = LoadLabelMap(category_path=object_category_path)

        ## Build model from pretrain model
        self.human_model = CDCLModel_Pb(model=get_testing_model_resnet101(), weight_path=human_model_path, gpus=gpus, scale=scale, input_pad=input_pad, prefix_name=prefix_name)
        self.object_model = DNNModel_Object(model_path=object_model_path)
        self.face_model = DNNModel_Pb(model_path=face_model_path, prefix_name=prefix_name)
        self.place_model = DNNModel_Pb(model_path=place_model_path, prefix_name=prefix_name)

        ## Get audio analysis thread.
        self.audio_thread = audio_thread

    def stop(self):
        ## Finishes the video recording therefore the thread too
        if self.open:
            ## Stop video capture thread
            self.open = False
            self.video_cap.stop()

    def start(self):
        """[Launches the video showing function and analysis function using a thread]
        """
        ## Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.
        ## Thread for Quardo W analysis.
        analysis_thread = threading.Thread(target=self.analysis_frame, daemon=True, args=())
        ## Thread for show frame which get by IpcamCapture
        show_thread = threading.Thread(target=self.show_frame, daemon=True, args=())

        ## Start show & analysis thread.
        analysis_thread.start()
        show_thread.start()
    
    def show_frame(self):
        """[Get the latest frame from IpcamCapture]
        """        
        while (self.open == True):
            frame = self.video_cap.getframe()
            
            if frame is not None:
                self.frame = frame
            else:
                pass

    def analysis_frame(self):
        """[Get Quadro W informtion for latest frame from IpcamCapture.]
        """        
        while (self.open == True):
            ## Get the latest list of frames.
            frames = self.video_cap.getframes()
            
            if len(frames) > 0:
                ## take the middle frame from frame list.
                self.analysis_frame = frames[len(frames)//2]

                ## Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.
                ## Thread for where recontition, who recontition, what recontition and when recontition.
                where_thread = threading.Thread(target=self.where_recontition, daemon=True, args=())
                who_thread = threading.Thread(target=self.who_recontition, daemon=True, args=())
                what_thread = threading.Thread(target=self.what_recontition, daemon=True, args=())
                when_thread = threading.Thread(target=self.when_recontition, daemon=True, args=())

                ## Start where recontition, who recontition, what recontition and when recontition thread.
                where_thread.start()
                who_thread.start()
                what_thread.start()
                when_thread.start()

                ## Wait for where recontition, who recontition, what recontition and when recontition thread to end.
                where_thread.join()
                when_thread.join()
                what_thread.join()
                who_thread.join()
                

                ## Get audio analysis.
                self.speak = self.audio_thread.speak
                self.speaking_word = self.audio_thread.speaking_word

                ## Use q table to predict action.
                qtable_path = os.path.join(self.qtable_path, self.who+"_qtable.csv")
                if os.path.isfile(qtable_path):
                    ## Get q table.
                    qtable = pd.read_csv(qtable_path, index_col=0)
                    current_main_state = self.when + "_" + self.where + "_" + self.what[0]
                    
                    main_states_rows = [ _ for _ in qtable.columns]
                    if current_main_state in main_states_rows:
                        state_action = qtable.loc[current_main_state].values
                        ## Get high award of action from award table.
                        maxaction_index = np.argmax(state_action)
                        todo = main_states_rows[maxaction_index]
                    else:
                        todo = "None"
                else:
                    todo = "None"

                self.result = [self.who, 
                               self.what[0], 
                               self.when, 
                               self.where, 
                               self.speak, 
                               self.current_time, 
                               self.accum_time,
                               self.human_result_score,
                               self.place_result_score,
                               self.speaking_word[0], ## n 
                               self.speaking_word[1], ## nr
                               self.speaking_word[2], ## nrfg *
                               self.speaking_word[3], ## nt
                               self.speaking_word[4], ## nz
                               self.speaking_word[5], ## ns
                               self.speaking_word[6], ## tg
                               self.speaking_word[7], ## v
                               self.speaking_word[8], ## df *
                               self.speaking_word[9], ## vg
                               self.speaking_word[10], ## d
                               self.speaking_word[11], ## t
                               self.cam,
                               todo]

                ## Save result to record-----------------------------------------------------------------------------
                record_csvpath = os.path.join(self.record_path, self.result[0]+"_record.csv")
                
                ## This person has stored Quadro W information.
                if os.path.exists(record_csvpath):
                    ## Get the past record.
                    with open(record_csvpath, "r") as f:
                        csv_r = csv.reader(f)
                        record_result = list(csv_r)

                    ## Find where to insert the new Quadro W information according accumlate time.
                    if len(record_result) > 1:
                        for i in range(1, len(record_result)):
                            if int(self.result[6]) < int(record_result[i][6]):
                                insert_index = i
                                break
                            else:
                                insert_index = None
                    elif len(record_result) == 1:
                        if int(self.result[6]) < int(record_result[1][6]):
                                insert_index = i
                                break
                        else:
                            insert_index = None
                    
                    ## If insert in the middle.
                    if insert_index is not None:
                        ## Avoid duplicate records
                        ## Find the Quadro W information of the same camera before the insert index
                        for i in range(insert_index, -1, -1):
                            if record_result[i][-1] == self.cam:
                                pre_record = record_result[i]
                                break
                            
                            if i == 0:
                                pre_record = ["None" for _ in range(23)]
                        ## Find the Quadro W information of the same camera after the insert index
                        for i in range(len(record_result)-2, insert_index, -1):
                            if record_result[i][-1] == self.cam:
                                next_record = record_result[i]
                                break
                            
                            if i == insert_index+1:
                                next_record = ["None" for _ in range(23)]

                        ## Avoid duplicate records
                        ## Compare result with the record.(next reocord, previous record, next record with same cmaera, previous record with same camera)
                        for i in range(1, 5):
                            ## If have different, save it
                            if self.result[i] != record_result[insert_index][i] and self.result[i] != record_result[insert_index-1][i] \
                                                                                and self.result[i] != pre_record[i] \
                                                                                and self.result[i] != next_record[i]:
                                with open(record_csvpath, "r") as infile:
                                    reader = list(csv.reader(infile))
                                    reader.insert(insert_index, self.result[:-1])

                                with open(record_csvpath, "w" , newline='') as outfile:
                                    writer = csv.writer(outfile)
                                    for line in reader:
                                        writer.writerow(line)
                                break
                    ## Insert in last
                    else:
                        ## Avoid duplicate records
                        ## Find the Quadro W information of the same camera before the insert index
                        for i in range(len(record_result)-2, -1, -1):
                            if record_result[i][-1] == self.cam:
                                pre_record = record_result[i]
                                break

                            if i == 0:
                                pre_record = ["None" for _ in range(23)]

                        ## Compare result with the previous record and previous record with same camera.
                        for i in range(1, 5):
                            if self.result[i] != record_result[-1][i] and self.result[i] != pre_record[i]:
                                with open(record_csvpath, "a", newline='') as f:
                                    csv_w = csv.writer(f)
                                    csv_w.writerow(self.result[:-1])
                                break
                ## This person has not saved Quadro W information, save it.
                else:
                    with open(record_csvpath, "w", newline='') as f:
                        csv_w = csv.writer(f)
                        csv_w.writerow(["who_state", 
                                        "what_state", 
                                        "when_state", 
                                        "where_state", 
                                        "speak_state", 
                                        "current_time", 
                                        "accumlate_time",
                                        "who_state_score",
                                        "where_state_score",
                                        "n", 
                                        "nr",
                                        "nrfg",
                                        "nt",
                                        "nz",
                                        "ns",
                                        "tg",
                                        "v",
                                        "df",
                                        "vg",
                                        "d",
                                        "t",
                                        "camera"])
                        csv_w.writerow(self.result[:-1])
                
            else:
                pass

    def where_recontition(self):
        """[Recognize where is in frame.]
        """        
        ## Resize frame according to the input size of the model.
        place_frame = DLImgResize(self.analysis_frame, self.place_size[0], self.place_size[1])
        ## Recognize place.
        place_preds = self.place_model.predict(place_frame)
        
        ## Sort predict values from small to large with indice
        place_result = np.argsort(place_preds)
        ## Turn Sort result to scor
        self.place_result_score = Pro2Score(place_result, len(self.place_category))
        ## Sort predict values from large to small
        place_top_preds = np.argsort(place_preds)[::-1]

        self.where = self.place_category[place_top_preds[0]]

    def when_recontition(self):
        """[Recognize when is in frame.]
        """        
        now = datetime.now()
        self.current_time = now.strftime("%H:%M:%S")
        
        ## Count the time from the date 01/01/01 00:00:00
        day = now.toordinal()
        time = now.strftime("%H:%M:%S")
        self.accum_time = (day- 1)*60*60*24 + int(time.rsplit(":")[0])*60*60 + int(time.rsplit(":")[1])*60 + int(time.rsplit(":")[2])
        
        current_time_h = int(now.strftime("%H"))
        if (current_time_h > 6) and (current_time_h <= 12):
            self.when = "morning"
        elif (current_time_h > 12) and (current_time_h <= 18):
            self.when = "afternoon"
        elif (current_time_h > 18) and (current_time_h < 24):
            self.when = "evening"
        elif (current_time_h >= 0) and (current_time_h <= 6):
            self.when = "midnight"

    def who_recontition(self):
        """[Recognize who is in frame.]
        """        
        ## Resize frame according to the input size of the model.
        human_frame = ImgResize(self.analysis_frame, self.human_size[0], self.human_size[1])
        ## Segment human 
        seg_frame = self.human_model.predict(human_frame)
        ## Bounding box of head by Segment human 
        find_head = Detect(seg_frame)
        head_region = find_head.detectHead()
        
        ## If no head in frame
        if not(all(head_region)):
            human_result = [ 0 for _ in range(len(self.face_category))]
            self.who = "Nobody"
            self.human_result_score = [ 0 for _ in range(len(self.face_category))]
        else:
            ## Frame head region
            head = human_frame[head_region[0]:head_region[1], head_region[2]:head_region[3]]
            ## Resize frame according to the input size of the model.
            face_frame = DLImgResize(head, self.face_size[0], self.face_size[1])
            ## Recognize head 
            face_preds = self.face_model.predict(face_frame)

            ## Sort predict values from small to large with indice
            human_result = np.argsort(face_preds)
            ## Turn Sort result to score
            self.human_result_score = Pro2Score(human_result, len(self.face_category))
            ## Sort predict values from large to small
            face_top_preds = np.argsort(face_preds)[::-1]


            self.who = self.face_category[face_top_preds[0]]
        
        

    def what_recontition(self):
        """[Recognize what is in frame.]
        """        
        ## Detect what is in the screen---------------------------------------------------------------------------------------
        self.what = self.object_model.predict(self.analysis_frame, self.object_category)
