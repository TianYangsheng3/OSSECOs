import csv, os
import pymysql
from datetime import datetime, date, timedelta
from collections import namedtuple

#### "用于Gephi做动态网络的数据"

#### 得到PidSet中的项目形成节点表和边表
def get_data(FileRootpath, EndDate, PidSet, layer):
    NextPidSet = []
    header0 = ['id', 'label', 'owner_id', 'created_at', 'forked_from']
    Row = namedtuple('Row', header0)
    print("Layer "+str(layer)+"\t :", len(PidSet))
    
    if len(PidSet)>0:
        db = pymysql.connect(host='10.201.98.82', user='ystian', passwd='123456', db='ghtorrent_restore')
        cursor= db.cursor()
        NodeData = []
        EdgeData = []

        for pid in PidSet:
            sql = "select id,name,owner_id,created_at,forked_from  from projects where forked_from="\
                +str(pid)+ " and created_at< '" + EndDate.strftime('%Y-%m-%d') + "'"
            cursor.execute(sql)
            CurData = cursor.fetchall()
            for row in CurData:
                row = Row(*row)
                #### 将fork而来的项目放入下一层
                NextPidSet.append(row.id) 
                #### 添加Enddata到边表                      
                cur_edge = [row.id, row.forked_from, row.created_at, EndDate]
                EdgeData.append(cur_edge)        
                #### 添加EndData到节点表 
                cur_node = []   
                for v in row:
                    cur_node.append(v)
                cur_node.append(EndDate)
                NodeData.append(cur_node)
        print("Success to get data in Layer "+str(layer))
        db.close()
        filepath_node = FileRootpath + 'node.csv'
        filepath_edge = FileRootpath + 'edge.csv'

        to_file(filepath_node, NodeData, layer)
        to_file(filepath_edge, EdgeData, layer)

        if len(NextPidSet)>0:
            get_data(FileRootpath, EndDate, NextPidSet, layer+1)

#### 得到初始时的节点表（不需要得到边表）
def get_start_pids(FileRootpath, EndDate, PidSet):
    if len(PidSet)>0:
        db = pymysql.connect(host='10.201.98.82', user='ystian', passwd='123456', db='ghtorrent_restore')
        cursor= db.cursor()
        NodeData = []
        
        for pid in PidSet:
            sql = "select id,name,owner_id,created_at,forked_from  from projects where id="\
                +str(pid)+ " and created_at< '" + EndDate.strftime('%Y-%m-%d') + "'"
            cursor.execute(sql)
            Data = cursor.fetchall()
            for row in Data:
                cur_node = []   
                for v in row:
                    cur_node.append(v)
                cur_node.append(EndDate)
                NodeData.append(cur_node)
        filepath = FileRootpath + 'node.csv'
        to_file(filepath, NodeData, -1)

def to_file(filepath, data, layer):
    isOK = False
    with open(filepath, 'a+', newline='') as f:
        f_csv = csv.writer(f)
        # f_csv.writerow(header)
        for row in data:
            f_csv.writerow(row)
        isOK = True 
    if isOK:
        print('Layer '+str(layer)+':\t Success to write to file: '+filepath)
    else:
        print('Layer '+str(layer)+':\t Failed to write to file: '+filepath)

#### 开始收集数据前的准备，主要是创建node.csv   edge.csv两个表
def prepare(FileRootpath):
    header_node = ('id', 'label', 'owner_id', 'created_at', 'forked_from','end_date')
    header_edge = ('source', 'target', 'created_at','end_date')
    file_edge = FileRootpath + 'edge.csv'
    file_node = FileRootpath + 'node.csv'    
    with open(file_node, 'a+', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header_node)
    with open(file_edge, 'a+', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header_edge)

if __name__ == '__main__':

    FileRootpath = 'data\\'
    EndDate = datetime(2018, 6, 1)
    prepare(FileRootpath)
    # print(EndDate)
    PidSet = [4023, 4485, 961]
    get_start_pids(FileRootpath, EndDate, PidSet)
    get_data(FileRootpath, EndDate, PidSet, 0)

