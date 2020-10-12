import cv2
import time
import threading
import os

class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False

        self.capture = cv2.VideoCapture(URL)

    def start(self):
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()

URL = 'rtsp://uscclab:uscclab@192.168.65.66:554/stream1'
i = 0

save_root_path = "/home/uscc/0808data/39/"
if not(os.path.exists(save_root_path)):
    os.mkdir(save_root_path)


ipcam = ipcamCapture(URL)
ipcam.start()
time.sleep(1)

while True:
    frame = ipcam.getframe()
    path = os.path.join(save_root_path, str(i)+".jpg")
    # cv2.imwrite(path, frame)

    i+=1
    
    cv2.imshow('Image', frame)
    if cv2.waitKey(1000) == 27:
        cv2.destroyAllWindows()
        ipcam.stop()
        break
