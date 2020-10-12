import sys
import cv2
import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from video_recorder import VideoRecorder
from audio_recorder import AudioRecorder
from access_QWrecord import RecordUpdata
from qlearning import QLearningUpdata

from qt_ui.main_ui import Ui_dialog
from qt_ui.speach_ui import Ui_dialog_Speaching

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_dialog()
        self.ui.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.onBindingUI()

        ##
        self.record_header_labels = ['who', 'what', 'when', 'where', 'speaking']
        
        ## Create detail speaking information window
        self.speaking_ui_cam1 = SpeakingMainWindow()
        self.speaking_ui_cam2 = SpeakingMainWindow()

        ## For audio and video analysis from camera
        global video_thread_cam1
        global video_thread_cam2
        global audio_thread_cam1
        global audio_thread_cam2

        ## For showing Quadro W record
        global nobody_record
        global chung_chih_record
        global yu_sung_record
        global chang_yu_record
        global i_hsin_record
        global tzu_yuan_record
        global chih_yu_record
        global other_record

        ## For Q Learing
        global nobody_q_table
        global chung_chih_q_table
        global yu_sung_q_table
        global chang_yu_q_table
        global i_hsin_q_table
        global tzu_yuan_q_table
        global other_q_table

        ## Create video and audio proces thread and process them to get Quadro W info
        audio_thread_cam1 = AudioRecorder(cam_ip='rtsp://uscclab:uscclab@192.168.65.66:554/stream1',
                                          audio_spath='temporarily_rtsp_data/for_speaking/cam1',
                                          audio_stime=5,
                                          pos_path='nlp_recognize/pos.json',
                                          dict_path='nlp_recognize/extra_dict.txt')
        video_thread_cam1 = VideoRecorder(cam_ip='rtsp://uscclab:uscclab@192.168.65.66:554/stream1', 
                                          record_path='record/',
                                          qtable_path='q_table/',
                                          compress_result=True,
                                          fps=15, 
                                          analysis_sec=1, 
                                          prefix_name="cam1",
                                          audio_thread=audio_thread_cam1,
                                          human_model_path="trans_model/model_simulated_RGB_mgpu_scaling_append.0024.pb",
                                          face_model_path='trans_model/face_new3.pb',
                                          place_model_path='trans_model/place_new3.pb',
                                          object_model_path='object_recognize/code/workspace/training_demo/model/pb/frozen_inference_graph.pb',
                                          face_category_path='face_recognize/categories_human_uscc.txt', 
                                          place_category_path='places_recognize/categories_places_uscc.txt',
                                          object_category_path='object_recognize/code/workspace/training_demo/annotations/label_map.pbtxt')
        audio_thread_cam2 = AudioRecorder(cam_ip='rtsp://uscclab:uscclab@192.168.65.41:554/stream1',
                                          audio_spath='temporarily_rtsp_data/for_speaking/cam2',
                                          audio_stime=5,
                                          pos_path='nlp_recognize/pos.json',
                                          dict_path='nlp_recognize/extra_dict.txt')
        video_thread_cam2 = VideoRecorder(cam_ip='rtsp://uscclab:uscclab@192.168.65.41:554/stream1', 
                                          record_path='record/',
                                          qtable_path='q_table/',
                                          compress_result=True,
                                          fps=15, 
                                          analysis_sec=1, 
                                          prefix_name="cam2",
                                          audio_thread=audio_thread_cam2,
                                          human_model_path="trans_model/model_simulated_RGB_mgpu_scaling_append.0024.pb",
                                          face_model_path='trans_model/face_new3.pb',
                                          place_model_path='trans_model/place_new3.pb',
                                          object_model_path='object_recognize/code/workspace/training_demo/model/pb/frozen_inference_graph.pb',
                                          face_category_path='face_recognize/categories_human_uscc.txt', 
                                          place_category_path='places_recognize/categories_places_uscc.txt',
                                          object_category_path='object_recognize/code/workspace/training_demo/annotations/label_map.pbtxt')

        ## Create thread to show record of Quadro W info
        nobody_record = RecordUpdata(csv_path="record/Nobody_record.csv")
        chung_chih_record = RecordUpdata(csv_path="record/chung-chih_record.csv")
        yu_sung_record = RecordUpdata(csv_path="record/yu-sung_record.csv")
        chang_yu_record = RecordUpdata(csv_path="record/chang-yu_record.csv")
        i_hsin_record = RecordUpdata(csv_path="record/i-hsin_record.csv")
        tzu_yuan_record = RecordUpdata(csv_path="record/tzu-yuan_record.csv")
        chih_yu_record = RecordUpdata(csv_path="record/chih-yu_record.csv")

        ## Create thread to calculate Q value and show
        nobody_q_table = QLearningUpdata(in_record_path="record/Nobody_record.csv", 
                                         out_table_path="q_table/Nobody_qtable.csv", 
                                         where_pool=[[2], [1], []],
                                         where_category_path="places_recognize/categories_places_uscc.txt",
                                         care_number=100, 
                                         decay_reward=0.98, 
                                         base_reward=100, 
                                         lower_limit=1, 
                                         decay_qvalue=0.9, 
                                         learning_rate=0.1)
        chung_chih_q_table = QLearningUpdata(in_record_path="record/chung-chih_record.csv", 
                                             out_table_path="q_table/chung-chih_qtable.csv", 
                                             where_pool=[[2], [1], []],
                                             where_category_path="places_recognize/categories_places_uscc.txt",
                                             care_number=100, 
                                             decay_reward=0.98, 
                                             base_reward=100, 
                                             lower_limit=1, 
                                             decay_qvalue=0.9, 
                                             learning_rate=0.1)
        yu_sung_q_table = QLearningUpdata(in_record_path="record/yu-sung_record.csv", 
                                          out_table_path="q_table/yu-sung_qtable.csv", 
                                          where_pool=[[2], [1], []],
                                          where_category_path="places_recognize/categories_places_uscc.txt",
                                          care_number=100, 
                                          decay_reward=0.98, 
                                          base_reward=100, 
                                          lower_limit=1, 
                                          decay_qvalue=0.9, 
                                          learning_rate=0.1)
        chang_yu_q_table = QLearningUpdata(in_record_path="record/chang-yu_record.csv", 
                                           out_table_path="q_table/chang-yu_qtable.csv", 
                                           where_pool=[[2], [1], []],
                                           where_category_path="places_recognize/categories_places_uscc.txt",
                                           care_number=100, 
                                           decay_reward=0.98, 
                                           base_reward=100, 
                                           lower_limit=1, 
                                           decay_qvalue=0.9, 
                                           learning_rate=0.1)
        i_hsin_q_table = QLearningUpdata(in_record_path="record/i-hsin_record.csv", 
                                         out_table_path="q_table/i-hsin_qtable.csv", 
                                         where_pool=[[2], [1], []],
                                         where_category_path="places_recognize/categories_places_uscc.txt",
                                         care_number=100, 
                                         decay_reward=0.98, 
                                         base_reward=100, 
                                         lower_limit=1, 
                                         decay_qvalue=0.9, 
                                         learning_rate=0.1)
        tzu_yuan_q_table = QLearningUpdata(in_record_path="record/tzu-yuan_record.csv", 
                                           out_table_path="q_table/tzu-yuan_qtable.csv", 
                                           where_pool=[[2], [1], []],
                                           where_category_path="places_recognize/categories_places_uscc.txt",
                                           care_number=100,
                                           decay_reward=0.98,
                                           base_reward=100,
                                           lower_limit=1,
                                           decay_qvalue=0.9,
                                           learning_rate=0.1)
        chih_yu_q_table = QLearningUpdata(in_record_path="record/chih-yu_record.csv", 
                                          out_table_path="q_table/chih-yu_qtable.csv", 
                                          where_pool=[[2], [1], []],
                                          where_category_path="places_recognize/categories_places_uscc.txt",
                                          care_number=100, 
                                          decay_reward=0.98, 
                                          base_reward=100, 
                                          lower_limit=1, 
                                          decay_qvalue=0.9, 
                                          learning_rate=0.1)
       
        ## Start all thread
        video_thread_cam1.start()
        video_thread_cam2.start()
        audio_thread_cam1.start()
        audio_thread_cam2.start()

        nobody_record.start()
        chung_chih_record.start()
        yu_sung_record.start()
        chang_yu_record.start()
        i_hsin_record.start()
        tzu_yuan_record.start()
        chih_yu_record.start()

        nobody_q_table.start()
        chih_yu_q_table.start()
        yu_sung_q_table.start()
        chang_yu_q_table.start()
        i_hsin_q_table.start()
        tzu_yuan_q_table.start()
        chang_yu_q_table.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_webcam)
        self.timer.start(0)

    def onBindingUI(self):
        self.ui.pushButton.clicked.connect(self.click_button_cam1)
        self.ui.pushButton_2.clicked.connect(self.click_button_cam2)

    def click_button_cam1(self):
        self.speaking_ui_cam1.show()

    def click_button_cam2(self):
        self.speaking_ui_cam2.show()

    def start_webcam(self):
        ## -----------------------------------------------------------------------------------------------------------------
        ## Show Cam1 information on window
        frame = video_thread_cam1.frame
        frame = cv2.resize(frame, (1152, 631))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame,
                       frame.shape[1],
                       frame.shape[0],
                       frame.shape[1] * 3,
                       QImage.Format_RGB888)
        self.ui.frame1_frame.setPixmap(QPixmap.fromImage(image))
        self.ui.frame1_when.setText(video_thread_cam1.result[5])
        self.ui.frame1_where.setText(video_thread_cam1.result[3])
        self.ui.frame1_who.setText(video_thread_cam1.result[0])
        self.ui.frame1_what.setText(video_thread_cam1.result[1])
        self.ui.frame1_hearing.setText(video_thread_cam1.result[4])
        self.ui.frame1_todo.setText(video_thread_cam1.result[22].replace("_", "\n"))

        ## Show cam2 information on window
        frame = video_thread_cam2.frame
        frame = cv2.resize(frame, (1152, 631))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame,
                       frame.shape[1],
                       frame.shape[0],
                       frame.shape[1] * 3,
                       QImage.Format_RGB888)
        self.ui.frame2_frame.setPixmap(QPixmap.fromImage(image))
        self.ui.frame2_when.setText(video_thread_cam2.result[5])
        self.ui.frame2_where.setText(video_thread_cam2.result[3])
        self.ui.frame2_who.setText(video_thread_cam2.result[0])
        self.ui.frame2_what.setText(video_thread_cam2.result[1])
        self.ui.frame2_hearing.setText(video_thread_cam2.result[4])
        self.ui.frame2_todo.setText(video_thread_cam1.result[22].replace("_", "\n"))

        ## -----------------------------------------------------------------------------------------------------------------
        # Show cam1 detail speaking information
        for i in range(len(video_thread_cam1.speaking_word)):
            for j in range(len(video_thread_cam1.speaking_word[i])):
                self.speaking_ui_cam1.tableWidget.setItem(i, j, QTableWidgetItem(video_thread_cam1.speaking_word[i][j]))

        ## Show cam2 detail speaking information
        for i in range(len(video_thread_cam2.speaking_word)):
            for j in range(len(video_thread_cam2.speaking_word[i])):
                self.speaking_ui_cam2.tableWidget.setItem(i, j, QTableWidgetItem(video_thread_cam2.speaking_word[i][j]))

        ## -----------------------------------------------------------------------------------------------------------------
        ## Show nobody record of Quardo W info
        table_size, record = nobody_record.getdatae()
        self.ui.tableWidget.setRowCount(table_size[0]-2)
        self.ui.tableWidget.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget.resizeColumnsToContents()
        self.ui.tableWidget.resizeRowsToContents()

        ## Show chung_chih record of Quardo W info
        table_size, record = chung_chih_record.getdatae()
        self.ui.tableWidget_2.setRowCount(table_size[0]-2)
        self.ui.tableWidget_2.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_2.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_2.resizeColumnsToContents()
        self.ui.tableWidget_2.resizeRowsToContents()

        ## Show yu_sung record of Quardo W info
        table_size, record = yu_sung_record.getdatae()
        self.ui.tableWidget_3.setRowCount(table_size[0]-2)
        self.ui.tableWidget_3.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_3.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_3.resizeColumnsToContents()
        self.ui.tableWidget_3.resizeRowsToContents()

        ## Show chang_yu record of Quardo W info
        table_size, record = chang_yu_record.getdatae()
        self.ui.tableWidget_4.setRowCount(table_size[0]-2)
        self.ui.tableWidget_4.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_4.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_4.resizeColumnsToContents()
        self.ui.tableWidget_4.resizeRowsToContents()

        ## Show i_hsin record of Quardo W info
        table_size, record = i_hsin_record.getdatae()
        self.ui.tableWidget_5.setRowCount(table_size[0]-2)
        self.ui.tableWidget_5.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_5.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_5.resizeColumnsToContents()
        self.ui.tableWidget_5.resizeRowsToContents()

        ## Show tzu_yuan record of Quardo W info
        table_size, record = tzu_yuan_record.getdatae()
        self.ui.tableWidget_6.setRowCount(table_size[0]-2)
        self.ui.tableWidget_6.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_6.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_6.resizeColumnsToContents()
        self.ui.tableWidget_6.resizeRowsToContents()

        ## Show chih_yu record of Quardo W info
        table_size, record = chih_yu_record.getdatae()
        self.ui.tableWidget_7.setRowCount(table_size[0]-2)
        self.ui.tableWidget_7.setColumnCount(table_size[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(self.record_header_labels)

        for i, row in enumerate(record):
            if i == 0:
                continue
            for j, string in enumerate(record[i]):
                self.ui.tableWidget_7.setItem(i-1, j, QTableWidgetItem(string))
        self.ui.tableWidget_7.resizeColumnsToContents()
        self.ui.tableWidget_7.resizeRowsToContents()

        ## -----------------------------------------------------------------------------------------------------------------
        ## Show nobody reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = nobody_q_table.getdata()
        
        self.ui.tableWidget_16.setRowCount(reward_table_size[0])
        self.ui.tableWidget_16.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_16.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_16.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_16.setItem(i, j, string)
        self.ui.tableWidget_16.resizeColumnsToContents()
        self.ui.tableWidget_16.resizeRowsToContents()

        self.ui.tableWidget_24.setRowCount(q_table_size[0])
        self.ui.tableWidget_24.setColumnCount(q_table_size[1])
        self.ui.tableWidget_24.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_24.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_24.setItem(i, j, string)
        self.ui.tableWidget_24.resizeColumnsToContents()
        self.ui.tableWidget_24.resizeRowsToContents()

        # Show chih_yu reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = chang_yu_q_table.getdata()
        
        self.ui.tableWidget_17.setRowCount(reward_table_size[0])
        self.ui.tableWidget_17.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_17.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_17.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_17.setItem(i, j, string)
        self.ui.tableWidget_17.resizeColumnsToContents()
        self.ui.tableWidget_17.resizeRowsToContents()

        self.ui.tableWidget_25.setRowCount(q_table_size[0])
        self.ui.tableWidget_25.setColumnCount(q_table_size[1])
        self.ui.tableWidget_25.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_25.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_25.setItem(i, j, string)
        self.ui.tableWidget_25.resizeColumnsToContents()
        self.ui.tableWidget_25.resizeRowsToContents()

        # Show yu_sung reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = yu_sung_q_table.getdata()
        
        self.ui.tableWidget_18.setRowCount(reward_table_size[0])
        self.ui.tableWidget_18.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_18.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_18.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_18.setItem(i, j, string)
        self.ui.tableWidget_18.resizeColumnsToContents()
        self.ui.tableWidget_18.resizeRowsToContents()

        self.ui.tableWidget_26.setRowCount(q_table_size[0])
        self.ui.tableWidget_26.setColumnCount(q_table_size[1])
        self.ui.tableWidget_26.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_26.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_26.setItem(i, j, string)
        self.ui.tableWidget_26.resizeColumnsToContents()
        self.ui.tableWidget_26.resizeRowsToContents()

        # Show chang_yu reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = chang_yu_q_table.getdata()
        
        self.ui.tableWidget_19.setRowCount(reward_table_size[0])
        self.ui.tableWidget_19.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_19.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_19.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_19.setItem(i, j, string)
        self.ui.tableWidget_19.resizeColumnsToContents()
        self.ui.tableWidget_19.resizeRowsToContents()

        self.ui.tableWidget_27.setRowCount(q_table_size[0])
        self.ui.tableWidget_27.setColumnCount(q_table_size[1])
        self.ui.tableWidget_27.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_27.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_27.setItem(i, j, string)
        self.ui.tableWidget_27.resizeColumnsToContents()
        self.ui.tableWidget_27.resizeRowsToContents()

        # Show i_hsin reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = i_hsin_q_table.getdata()
        
        self.ui.tableWidget_20.setRowCount(reward_table_size[0])
        self.ui.tableWidget_20.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_20.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_20.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_20.setItem(i, j, string)
        self.ui.tableWidget_20.resizeColumnsToContents()
        self.ui.tableWidget_20.resizeRowsToContents()

        self.ui.tableWidget_28.setRowCount(q_table_size[0])
        self.ui.tableWidget_28.setColumnCount(q_table_size[1])
        self.ui.tableWidget_28.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_28.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_28.setItem(i, j, string)
        self.ui.tableWidget_28.resizeColumnsToContents()
        self.ui.tableWidget_28.resizeRowsToContents()

        # Show tzu_yuan reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = tzu_yuan_q_table.getdata()
        
        self.ui.tableWidget_21.setRowCount(reward_table_size[0])
        self.ui.tableWidget_21.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_21.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_21.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_21.setItem(i, j, string)
        self.ui.tableWidget_21.resizeColumnsToContents()
        self.ui.tableWidget_21.resizeRowsToContents()

        self.ui.tableWidget_29.setRowCount(q_table_size[0])
        self.ui.tableWidget_29.setColumnCount(q_table_size[1])
        self.ui.tableWidget_29.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_29.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_29.setItem(i, j, string)
        self.ui.tableWidget_29.resizeColumnsToContents()
        self.ui.tableWidget_29.resizeRowsToContents()

        # Show chang_yu reward table and q table
        reward_table_size, reward_table_header, reward_table, q_table_size, qtable_table_header, q_table = chang_yu_q_table.getdata()
        
        self.ui.tableWidget_22.setRowCount(reward_table_size[0])
        self.ui.tableWidget_22.setColumnCount(reward_table_size[1])
        self.ui.tableWidget_22.setHorizontalHeaderLabels(reward_table_header)
        self.ui.tableWidget_22.setVerticalHeaderLabels(reward_table_header)

        for i, row in enumerate(reward_table):
            for j, string in enumerate(reward_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_22.setItem(i, j, string)
        self.ui.tableWidget_22.resizeColumnsToContents()
        self.ui.tableWidget_22.resizeRowsToContents()

        self.ui.tableWidget_30.setRowCount(q_table_size[0])
        self.ui.tableWidget_30.setColumnCount(q_table_size[1])
        self.ui.tableWidget_30.setHorizontalHeaderLabels(qtable_table_header)
        self.ui.tableWidget_30.setVerticalHeaderLabels(qtable_table_header)

        for i, row in enumerate(q_table):
            for j, string in enumerate(q_table[i]):
                string = QtWidgets.QTableWidgetItem(str(string))
                self.ui.tableWidget_30.setItem(i, j, string)
        self.ui.tableWidget_30.resizeColumnsToContents()
        self.ui.tableWidget_30.resizeRowsToContents()

class SpeakingMainWindow(QWidget, Ui_dialog_Speaching):
    def __init__(self, parent=None):
        super(SpeakingMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())

        self.pushButton.clicked.connect(self.close)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())