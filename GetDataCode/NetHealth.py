import csv, os, json
import numpy as np
from collections import namedtuple
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import PCA
import pymysql
import matplotlib.pyplot as plt

#### 根据中心项目的历史数据获得每个月的每个指标的权重
def GetIndicatorWeights(FileRootpath, FinalNetIds, TimeLens, batch_size):
    header = ['date', 'pid', 'created_at', 'forked_from', 'forks','contributor','commits','commit_comment',\
        'req_opened','req_closed','req_merged','req_other','issue','issue_comment','watchers', 'in_degree', 'layer', 'brothernum']
    Row = namedtuple('Row', header)

    Data = [[] for _ in range(TimeLens)]
    for netid in FinalNetIds:
        filepath = FileRootpath + "Net_Data_"+str(netid)+".csv"
        with open(filepath, 'r') as f:
            f_csv = csv.reader(f)
            next(f_csv)
            cnt = 0
            for row in f_csv:
                cur_data = []
                if cnt%batch_size==0:
                    if cnt/batch_size>=TimeLens:
                        break
                    for i in range(4, 15):
                        cur_data.append(float(row[i]))
                    Data[int(cnt/batch_size)].append(cur_data)
                cnt += 1
    Weights = []
    MinMAx = []                

    for cur in range(TimeLens):
        curData = np.array(Data[cur])
        curMax = curData.max(axis=0)
        curMin = curData.min(axis=0)
        MinMAx.append([curMax, curMin])
        tmp = curMax-curMin
        for i in range(len(tmp)):
            if tmp[i] == 0:
                tmp[i] = 1
        curData = (curData - curMin)/tmp
        # pca = PCA(n_components='mle')
        pca = PCA(n_components=3)
        pca.fit(curData)
        w = pca.explained_variance_ratio_
        components = pca.components_
        indi_w = np.dot(w, components)
        # indiw_sum = indi_w.sum()
        # print("indi_w: ", indi_w)
        # print("最大值是：",indi_w.max())
        # print("最小值是：",indi_w.min())
        for j in range(len(indi_w)):
            indi_w[j] = indi_w[j]/indi_w.sum()
            # if indi_w[j]<0:
            #     print("出现负数权重！")
        Weights.append(indi_w)
    return Weights, MinMAx


            
def ComputeCentrality(Layers, NodeSum, Pid, PidLayer, PidBrotherNums, PidSonNums):
    dis = 0
    for j in range(len(Layers)):
        cur_layer_nums = Layers[j]
        if j==PidLayer:
            cur_layer_nums -= (PidBrotherNums +1)
            dis += (2*PidBrotherNums)
        elif j==PidLayer+1:
            cur_layer_nums -= PidSonNums
            dis += PidSonNums
        elif j==PidLayer-1:
            cur_layer_nums -=1
            dis += 1
        dis += (PidLayer + j)*cur_layer_nums
    if dis==0:
        dis = 1
    cen = (NodeSum-1)/dis
    return cen


#### 得到网络NetId的健康性分值，从created到2018-06-01
def GetNetHealth(FileRootpath, NetId, TimeLens, batch_size, Weights, MinMax):
    filepath_layerdata = FileRootpath + 'Net_'+str(NetId[0])+'_LayerDataIncre.csv'
    Layers = []
    with open(filepath_layerdata, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            cur = [int(row[i]) for i in range(1, 8)]
            Layers.append(cur)

    filepath_netdata = FileRootpath + 'Net_Data_' + str(NetId[0]) + '.csv'
    NetHealth = []
    with open(filepath_netdata, 'r') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        Row = namedtuple('Row', header)
        cnt, ind, seq = 0, 0, 0
        cur_nethealth = 0
        for row in f_csv:
            row = Row(*row)
            if cnt%batch_size==0:
                cur_date = row[0]
                cur_nethealth = 0
                ind = int(cnt/batch_size) 
                seq = int(cnt/batch_size)
                if ind>=TimeLens:
                    ind = TimeLens-1
            #### 求项目pid的健康性    
            pid = int(row[1])
            cur_tmp = MinMax[ind][0] - MinMax[ind][1]
            for j in range(len(cur_tmp)):
                if cur_tmp[j]==0:
                    cur_tmp[j] = 1
            cur_piddata = [float(row[i]) for i in range(4, 15)]
            cur_piddata = np.asarray(cur_piddata)
            cur_piddata = (cur_piddata - MinMax[ind][1])/cur_tmp
            for k in range(len(cur_piddata)):
                if cur_piddata[k]<0:
                    cur_piddata[k] = 0
            cur_pidhealth = np.dot(Weights[ind], cur_piddata)
            cur_layer = Layers[seq][1:]
            c = ComputeCentrality(cur_layer, Layers[seq][0], pid, int(row.layer), int(row.brothernum), int(row.in_degree))

            cur_nethealth += c*cur_pidhealth
            cnt += 1
            if cnt%batch_size==0:
                NetHealth.append([cur_date, cur_nethealth])
    filepath = FileRootpath + 'Net_'+str(NetId[0])+'_health.csv'
    ToFile(filepath, NetHealth)
    print('Success write health of Net: %d to file !!!' %NetId[0])


def ToFile(filepath, data):
    tmp = []
    with open(filepath, 'w', newline="") as f:
        f_csv = csv.writer(f)
        for row in data:
            f_csv.writerow(row)
            tmp.append(row[1])
    x = range(len(tmp))
    plt.plot(x, tmp)
    plt.show()

if __name__ == '__main__':
    FileRootpath = 'GetDataCode\\data\\'
    EndDate = datetime(2018, 6, 1)
    AdjSize = 1000
    TimeLens = 73
    FinalNetIds = [418, 866, 1115, 1282]
    Weights, MaxMin = GetIndicatorWeights(FileRootpath, FinalNetIds, TimeLens, AdjSize)
    for netid in FinalNetIds:
        GetNetHealth(FileRootpath, [netid], TimeLens, AdjSize, Weights, MaxMin)
    

