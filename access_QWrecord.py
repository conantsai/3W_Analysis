import threading
import csv
import os
from PyQt5.QtWidgets import QTableWidgetItem

class RecordUpdata:
    """[Get Quadro W information.]
    """    
    def __init__(self, csv_path):
        """[Initial RecordUpdate class.]

        Args:
            csv_path ([str]): [Path of Quadro W information record]
        """        
        ## Init setting
        self.table_size = [0, 0]
        self.table_record = list(list())
        self.csv_path = csv_path
        self.isstop = False

    def start(self):
        """[Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.]
        """        
        print("get record started!")
        threading.Thread(target=self.querydata, daemon=True, args=()).start()

    def stop(self):
        """[Switch to stop infinite loop.]
        """        
        self.isstop = True
        print("get stopped!")
   
    def getdatae(self):
        """[When there is a need for record, the latest record is returned.]
        """        
        return self.table_size, self.table_record
        
    def querydata(self):
        """[Get latest record]
        """        
        while (not self.isstop):
            if os.path.exists(self.csv_path):
                ## Create table.
                with open(self.csv_path, newline="") as fileInput:
                    rows = csv.reader(fileInput)
                    col_count = 5  ## Number of rows to show
                    row_count = 1
                    for row in rows:
                        row_count = row_count + 1
                table_record = [[None for i in range(col_count)] for j in range(row_count)]
                
                ## Put the record into table
                with open(self.csv_path, newline="") as fileInput:
                    rows = csv.reader(fileInput)
                    for i, row in enumerate(rows):
                        for j, string in enumerate(row):
                            if j > 4: ## Number of rows to show
                                break
                            ########################################可能有錯
                            try:
                                table_record[i][j] = string
                            except IndexError:
                                print(table_record, i, j)
                                pass
                self.table_record = table_record
                self.table_size = [row_count, col_count]
            else:
                self.table_record = list(list())
                self.table_size = [0, 0]