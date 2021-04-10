import csv, os
import json
import pymysql
from datetime import datetime, date, timedelta
from collections import namedtuple
from dateutil.relativedelta import relativedelta


#### "用于图神经网络训练的数据"

#### 递归得到网络Pid中的项目节点
def GetNetNodes(FileRootpath, EndDate, rootPid, Pid, Nodes, LayerNodes, layer):
    NextPids = []
    header0 = ['id', 'label', 'owner_id', 'created_at', 'forked_from']
    Row = namedtuple('Row', header0)
    # print("Layer "+str(layer)+"\t :", len(Pid))
    
    db = pymysql.connect(host='10.201.98.82', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor= db.cursor()
    for pid in Pid:
        sql = "select id,name,owner_id,created_at,forked_from  from projects where forked_from="\
            +str(pid)+ " and created_at< '" + EndDate.strftime('%Y-%m-%d') + "'"
        cursor.execute(sql)
        CurData = cursor.fetchall()
        for row in CurData:
            row = Row(*row)
            #### 将fork而来的项目放入下一层
            NextPids.append(row.id) 
            cur_node = []   
            for v in row:
                cur_node.append(v)
            cur_node.append(EndDate)
            Nodes[row.id] = cur_node
    # print("Success to get data in Layer "+str(layer))
    db.close()

    if len(NextPids)>0:
        LayerNodes[rootPid][layer+1]=NextPids[:]
        GetNetNodes(FileRootpath, EndDate, rootPid, NextPids, Nodes, LayerNodes, layer+1)


#### 得到网络Pid（即Pidset）初始时的节点
def GetInitNode(FileRootpath, EndDate, PidSet, Nodes, LayerNodes):
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
                Nodes[pid] = cur_node
            LayerNodes[pid][0] = [pid]
        db.close()

#### 将网络Pid中的节点id映射成邻接矩阵索引
def IdToIndex(Pid, Nodes, AdjSize, LowBound):
    if len(Nodes)>AdjSize:
        print("网络"+str(Pid[0])+"的节点数超过AdjSize (Nodes: "+str(len(Nodes))+", AdjSize: "+str(AdjSize)+")")
        return False
    elif len(Nodes)<LowBound:
        print("网络"+str(Pid[0])+"的节点数少于LowBound (Nodes: "+str(len(Nodes))+", LowBound: "+str(LowBound)+")")
        return False
    else:
        ind = 0
        for pid in Nodes:
            Nodes[pid][0] = ind
            ind += 1
        return True


#### 判断在CurDate这个月及之前，项目(Pdate)是否创建了
def isOK(Pdate, CurDate):
    if (CurDate.month - Pdate.month)+(CurDate.year - Pdate.year)*12>=0:
        return True
    else:
        return False

#### 判断项目(Pdate)是否创建在CurDate这个月
def isBetween(Pdate, CurDate):
    if (CurDate.month - Pdate.month)==0 and (CurDate.year - Pdate.year)==0:
        return True
    else:
        return False

#### 按月从数据库中得到每个项目的特征数据
def GetNetData(FileRootpath, EndDate, NetId, Nodes, AdjSize):
    # header0 = ['id', 'label', 'owner_id', 'created_at', 'forked_from']
    # Row = namedtuple('Row', header0)

    db = pymysql.connect(host='10.201.98.82', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor= db.cursor()
    CreatDate = Nodes[NetId[0]][3]
    # print("CreatDate: ", CreatDate)
    StartDate = datetime(CreatDate.year, CreatDate.month, 1)
    # print("StartDate: ", StartDate)
    span = (EndDate.year-StartDate.year)*12 + (EndDate.month - StartDate.month)

    Data = []
    Adjancenies = []

    for i in range(span):
        cur_date = StartDate + relativedelta(months=i)
        next_date = cur_date + relativedelta(months=1)
        cur_month_data = [[] for _ in range(len(Nodes))]
        cur_month_adj = [0]*(AdjSize*AdjSize)
        #### 要存的数据 ['date', 'id', 'created_at', 'forked_from', 'forks','contributor','commits','commit_comment',
        #### 'req_opened','req_closed','req_merged','req_other','issue','issue_comment','watchers', 'in_degree', 'out_degree']
        for pid in Nodes:
            ind = Nodes[pid][0]
            if isOK(Nodes[pid][3], cur_date):
                #### 构建特征矩阵 shape：len(Nodes)*feature, 需要扩展成AdjSize*feature
                tmp = [cur_date, pid, Nodes[pid][3], Nodes[pid][4]]

                    #### 得到fork
                sql_fork = "select * from projects where forked_from=" + str(pid) + " and created_at>='"+cur_date.strftime('%Y-%m-%d')+\
                    "' and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_fork)
                data_fork = cursor.fetchall()
                tmp.append(round(len(data_fork)/30.0,6))
                # in_degree = len(data_fork)

                    #### 得到contributor and commits
                contributor = []
                contrib_num = 0
                sql_contrib = "select * from commits where project_id =" + str(pid) + " and created_at>='" + cur_date.strftime('%Y-%m-%d') +\
                     "' and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_contrib)
                data_contrib = cursor.fetchall()
                for row in data_contrib:
                    person_id = row[3]
                    if person_id not in contributor:
                        contrib_num += 1
                        contributor.append(person_id)
                tmp.append(round(contrib_num/30.0,6))
                tmp.append(round(len(data_contrib)/30.0, 6))

                    #### 得到commits comment
                sql_comment_c = "select * from (select * from commits where commits.project_id=" + str(pid) + ") as tmp "+ \
                    "join commit_comments on tmp.id=commit_comments.commit_id and commit_comments.created_at>='" +\
                        cur_date.strftime('%Y-%m-%d')+"' and commit_comments.created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_comment_c)
                data_comment_c = cursor.fetchall()
                tmp.append(round(len(data_comment_c)/30.0, 6))

                    #### 得到pull request
                sql_req = "select * from (select * from pull_requests where pull_requests.base_repo_id ="\
                    + str(pid) + ") as tmp join pull_request_history on tmp.id = pull_request_history.pull_request_id "+\
                        "  and pull_request_history.created_at>='"+ cur_date.strftime('%Y-%m-%d')+\
                            "' and pull_request_history.created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_req)
                data_req = cursor.fetchall()
                req_open, req_close, req_merge, req_other = 0, 0, 0, 0
                for row in data_req:
                    if row[10]=='opened':
                        req_open += 1
                    elif row[10]=='closed':
                        req_close += 1
                    elif row[10]=='merged':
                        req_merge += 1
                    else:
                        req_other += 1
                req = [round(req_open/30,6), round(req_close/30,6), round(req_merge/30,6), round(req_other/30,6)]
                tmp.extend(req)

                    #### 得到issue
                sql_issue = "select * from issues where repo_id=" + str(pid) + " and created_at>='"+cur_date.strftime('%Y-%m-%d')+\
                    "' and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"    
                cursor.execute(sql_issue)
                data_issue = cursor.fetchall()
                tmp.append(round(len(data_issue)/30.0,6))   

                    #### 得到issue comment
                sql_comment_i = "select * from (select * from issues where issues.repo_id=" + str(pid) + \
                    ") as tmp join issue_comments on tmp.id = issue_comments.issue_id and issue_comments.created_at>='"+\
                        cur_date.strftime('%Y-%m-%d')+ "' and issue_comments.created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_comment_i)
                data_comment_i = cursor.fetchall()
                tmp.append(round(len(data_comment_i)/30.0,6))    

                    #### 得到watcher
                sql_watcher =  "select * from watchers where repo_id = " + str(pid) +" and created_at>='"+ cur_date.strftime('%Y-%m-%d')+\
                    "' and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_watcher)
                data_watcher = cursor.fetchall()
                tmp.append(round(len(data_watcher)/30.0,6))

                    #### 入度（从初始到当月累计总数）, 入度即子女总数
                sql_indegree =  "select * from projects where forked_from=" + str(pid) +" and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                cursor.execute(sql_indegree)
                data_indegree = cursor.fetchall()
                tmp.append(len(data_indegree))
                
                    #### 节点所在层数
                pa_id = Nodes[pid][4]
                located = 0
                while pa_id != None:
                    located += 1    
                    pa_id = Nodes[pa_id][4]
                tmp.append(located)

                    #### 节点兄弟个数
                parent_id = Nodes[pid][4]
                if parent_id is None:
                    tmp.append(0)
                else:

                    sql_brothernum =  "select * from projects where forked_from=" + str(parent_id) +" and created_at<'"+next_date.strftime('%Y-%m-%d')+"'"
                    cursor.execute(sql_brothernum)
                    data_brothernum = cursor.fetchall()
                    tmp.append(len(data_brothernum)-1)

                cur_month_data[ind].extend(tmp)

                #### 构建邻接矩阵
                if Nodes[pid][4] is not None:
                    source_ind = ind
                    target_ind = Nodes[Nodes[pid][4]][0]
                    #### 暂时写成无向图
                    cur_month_adj[source_ind*AdjSize+target_ind] = 1
                    cur_month_adj[target_ind*AdjSize+source_ind] = 1
  
            else:
                tmp = [cur_date, pid, Nodes[pid][3], Nodes[pid][4]] + [0 for _ in range(14)]  ### 评价指标数量or特征数量 13
                cur_month_data[ind].extend(tmp)
        #### 将cur_month_data扩展到AdjSize*feature
        for _ in range(AdjSize-len(Nodes)):
            zeros = [0]*18
            cur_month_data.append(zeros)    
        Data.append(cur_month_data)
        Adjancenies.append(cur_month_adj)

    db.close()
    print("Get Net Data: ", str(NetId[0]))
    filepath_data = FileRootpath + "Net_Data_"+str(NetId[0])+".csv"
    filepath_adj = FileRootpath + "Net_Adj_"+str(NetId[0])+".csv"
    to_file(filepath_data, Data, NetId[0], True)
    to_file(filepath_adj, [Adjancenies], NetId[0], False)
    return Data, Adjancenies

#### 将Data, Adjancenies写到文件中
def to_file(filepath, data, NetId, flag):
    isOK = False
    header = ['date', 'pid', 'created_at', 'forked_from', 'forks','contributor','commits','commit_comment',\
        'req_opened','req_closed','req_merged','req_other','issue','issue_comment','watchers', 'in_degree', 'layer', 'brothernum']
    with open(filepath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        if flag:
            f_csv.writerow(header)
        for row in data:
            for row_i in row:
                f_csv.writerow(row_i)
        isOK = True 
    if isOK:
        print('NetId '+str(NetId)+':\t Success to write to file: '+filepath)
    else:
        print('NetId '+str(NetId)+':\t Failed to write to file: '+filepath)

#### 从validfile，即valid.csv 中得到中心项目的ID，每个中心项目将会形成一个网络Net   
def GetValidId(validfile):
	projects_valid = []
	with open(validfile, 'r') as f:
		f_csv = csv.reader(f)
		for row in f_csv:
			projects_valid.append([int(row[0])])
	return projects_valid

#### 得到每个中心项目形成的网络，即节点数据Data 和 邻接矩阵Adjancenies
def GetNets(FileRootpath, EndDate, AdjSize, projects_valid, LayerNum, LowBound):
    cnt = 0
    LayerNodes = {}     #### 记录每一层的节点ID
    for NetId in projects_valid:
        Nodes = {}
        LayerNodes[NetId[0]] = {}
        path = FileRootpath + "Net_"+str(NetId[0])
        GetInitNode(FileRootpath, EndDate, NetId, Nodes, LayerNodes)
        GetNetNodes(FileRootpath, EndDate, NetId[0], NetId, Nodes, LayerNodes, 0)
        # print("Nodes size: ", len(Nodes))
        LayerNodes[NetId[0]]['Nodes'] = len(Nodes)

        res = IdToIndex(NetId, Nodes, AdjSize,LowBound)
        if res:
            TimesLayerNodes(FileRootpath, EndDate, NetId, LayerNodes[NetId[0]], Nodes, AdjSize, LayerNum)
            cnt += 1
            Data, Adjancenies = GetNetData(FileRootpath, EndDate, NetId, Nodes, AdjSize)
    print("满足Adjsize的Net个数：", cnt)

    filepath = FileRootpath + 'NetLayer.json'
    with open(filepath, 'w', newline="") as f:
        json.dump(LayerNodes, f)
    print("Success write LayerNodes to json file!!")
        # return Data, Adjancenies


#### 将网络每层layer的节点数量变化写入文件
def TimesLayerNodes(FileRootpath, EndDate, NetId, LayerNode, Nodes, AdjSize, LayerNum):
    #### Nodes: ['id', 'label', 'owner_id', 'created_at', 'forked_from']
    #### LayerData: ['date', 'Nodes', 'layer_0', ..., 'layer_n']

    if len(LayerNode)-1>LayerNum:
        print("网络"+str(NetId[0])+"的层数大于设置的LayerNum：(NetLayer: "+str(len(LayerNode)-1)+", LayerNum: "+str(LayerNum)+")")
        return None
    if len(LayerNode)-1>3:
        print("网络"+str(NetId[0])+"的层数比较好：(NetLayer: "+str(len(LayerNode)-1)+", AdjSize: "+str(AdjSize)+")")
    CreatDate = Nodes[NetId[0]][3]
    StartDate = datetime(CreatDate.year, CreatDate.month, 1)
    span = (EndDate.year-StartDate.year)*12 + (EndDate.month - StartDate.month)

    LayerData = [[] for _ in range(span)]           #### 每月的每层增加的数量
    LayerDataIncre = [[] for _ in range(span)]                            #### 叠加的每月的每层数量
    for i in range(span):
        cur_date = StartDate + relativedelta(months=i)
        LayerData[i].append(cur_date)
        LayerDataIncre[i].append(cur_date)
        LayerData[i].extend([0]*(LayerNum+1))
        LayerDataIncre[i].extend([0]*(LayerNum+1))

    for layer in LayerNode:
        if layer!='Nodes':
            for pid in LayerNode[layer]:
                month_ind = (Nodes[pid][3].year - StartDate.year)*12+(Nodes[pid][3].month - StartDate.month)
                LayerData[month_ind][int(layer)+2] += 1     #### 该月网络layer层增加的节点个数
                LayerData[month_ind][1] += 1                #### 该月整个网络增加的节点个数
                for i in range(month_ind, span):
                    LayerDataIncre[i][int(layer)+2] += 1    #### 该月网络layer层总的节点个数
                    LayerDataIncre[i][1] += 1               #### 该月整个网络的节点个数
    
    filepath_layerdata = FileRootpath + "Net_"+str(NetId[0])+"_LayerData.csv"
    filepath_layerdataincre = FileRootpath + "Net_"+str(NetId[0])+"_LayerDataIncre.csv"
    header = ['date', 'Nodes'] + ['layer_'+str(i) for i in range(LayerNum)]
    with open(filepath_layerdata, 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(LayerData)
    # print("succeed write LayerData to file! ")

    with open(filepath_layerdataincre, 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(LayerDataIncre)
    # print("succeed write LayerDataIncre to file! ")
    

if __name__ == '__main__':

    FileRootpath = 'GetDataCode\\data\\'
    EndDate = datetime(2018, 6, 1)
    AdjSize = 1000
    LowBound = 200
    LayerNum = 6
    validpath = FileRootpath + "valid.csv"
    # projects_valid = GetValidId(validpath)
    projects_valid = [[4023],[6223]]
    GetNets(FileRootpath, EndDate, AdjSize, projects_valid, LayerNum, LowBound)

