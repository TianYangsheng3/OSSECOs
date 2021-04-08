import numpy as np
import torch,csv
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from collections import namedtuple 
import matplotlib.pyplot as plt

#####  综述“A Comprehensive Survey on Graph Neural Networks”中的   NN4G  模型(公式16)
####  模型  H^(k) = f(XW^(k) + AH^(k-1)Θ^(k))

class Aggregator(nn.Module):
    def __init__(self, origin_featurenum, input_dims, output_dims, use_bias = False):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias
        self.weight_1 = nn.Parameter(torch.Tensor(origin_featurenum, output_dims))     #### 即 模型中的W^(k)
        self.weight_2 = nn.Parameter(torch.Tensor(input_dims, output_dims))     #### 即 模型中的Θ^(k)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dims))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_1)
        nn.init.kaiming_uniform_(self.weight_2)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    #### 邻接矩阵Adjacency，shape：N*N(N为节点个数)， Last_Hidden即模型中的H^(k-1), shape: N*input_dims,即上一层的output_dims
    ####                                        不过，第0层的Last_Hidden就是NodesFeature
    #### NodesFeature, 即图的特征矩阵，shape：N*d  d = input_dims
    def forward(self, NodesFeature, Adjacency, Last_Hidden):
        X = torch.matmul(Last_Hidden, self.weight_2)  
        Aggre = torch.matmul(Adjacency, X) 
        Own = torch.matmul(NodesFeature, self.weight_1)
        Hidden =  Own + Aggre
        if self.use_bias:
            Hidden += self.bias          

        return Hidden                       #### shape: N*output_dims


class SpatialGCN(nn.Module):        ###  NN4G  模型
    #### input_dims, 输入的特征个数（即特征矩阵X的列数 也记为d）,
    #### output_dims， 即最终抽取出每个节点的特征数
    #### hidden_dims，即GCN中中每个隐藏层的节点数，这里 hidden_dims， 是一个list。 hidden_dims[0]是层数，hidden_dims[i]即第i层节点数量
    #### 如果没有全连接层，那么hidden_dims[-1] = output_dims，即最后一个隐藏层节点数量就是output_dims
    def __init__(self, input_dims, output_dims, hidden_dims, use_bias = True):  
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.gcn = []
        self.gcn.append(Aggregator(input_dims, input_dims, hidden_dims[1]))
        for i in range(1, hidden_dims[0]):
            self.gcn.append(Aggregator(input_dims, hidden_dims[i], hidden_dims[i+1]))             #### 如果没有全连接层，那么hidden_dims[2] = output_dims
    #### NodesFeature，网络节点特征矩阵，shape：N*d or  N*input_dims
    #### Adjacency， 图邻接矩阵，shape：N*N
    #### Last_Hidden，上一个隐藏层的节点特征矩阵，第0层时就是NodesFeature，shape_0：N*input_dims, shape_1: N*hidden_dims[1]
    #### 输出Hidden，即最后一个隐藏层的节点特征，shape：N*hidden_dims[-1] 或者N*output_dims
    def forward(self, NodesFeature, Adjacency, Last_Hidden):
        for layer in range(self.hidden_dims[0]):
            gcn = self.gcn[layer]
            # Last_Hidden = NodesFeature
            Hidden = gcn(NodesFeature, Adjacency, Last_Hidden)
            if layer==0:
                Hidden = F.relu(Hidden)
            Last_Hidden = Hidden
        return Hidden                           #### shape: N*output_dims

class LSTM(nn.Module):
    #### input_dims，输入数据的特征个数，因为这里LSTM的输入是SpatialGCN的输出，故 input_dims = sGcn.output_dims
    #### hidden_dims, 隐藏层的节点个数，也是LSTM的输出的维度
    #### batch_size， 每次训练的数据条数，在这里指的是网络节点个数N
    #### layers：LSTM的层数
    def __init__(self, input_dims, hidden_dims, batch_size, layers):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.layers = layers
        self.lstm = nn.LSTM(input_dims, hidden_dims, layers)
        
    def forward(self, x):               #### x is the inputs, x shape: seq_num*batch_size*input_dims
        h0 = torch.randn(self.layers, self.batch_size, self.hidden_dims)
        c0 = torch.randn(self.layers, self.batch_size, self.hidden_dims)
        out, (hn, cn) = self.lstm(x, (h0, c0))          ### out shape: seq_num*batch_size*hidden_dims  
        return out

class LstmGcnNet(nn.Module):
    #### seq_num：LSTM步长
    #### input_dims：输入的节点的特征数，即特征矩阵的列数
    #### output_dims：最终输出的特征数，这里暂定为8，即整个图的健康性+总节点数+每一层的节点数（7层）
    #### batch_size: 每次训练的数据条数，在这里指的是网络节点个数N,即batch_size=N
    def __init__(self, seq_num, input_dims, output_dims, hidden_dims_lstm, hidden_dims_gcn, batch_size, layers_lstm):
        super().__init__()
        self.seq_num = seq_num
        self.input_dims = input_dims
        self.hidden_dims_lstm = hidden_dims_lstm
        self.hidden_dims_gcn = hidden_dims_gcn
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.layers = layers_lstm
        self.sGcn = SpatialGCN(input_dims, hidden_dims_gcn[-1], hidden_dims_gcn)
        self.lstm = LSTM(hidden_dims_gcn[-1], hidden_dims_lstm, batch_size, layers_lstm)
        self.fc1 = nn.Linear(batch_size*hidden_dims_lstm, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_dims)
    
    #### NodesFeatures: seq_num步的特征矩阵，shape：seq_num*N*input_dims  or seq_num*batch_size*input_dims
    def forward(self, NodesFeatures, Adjacencies):
        for i in range(self.seq_num):
            cur = self.sGcn(NodesFeatures[i], Adjacencies[i], NodesFeatures[i])     #### cur  shape: N*hidden_dims_gcn[-1] 0r N*sGCN_output_dims
            cur = cur.view(-1, self.batch_size, self.hidden_dims_gcn[-1])
            if i==0:
                sgcn_outs = cur
            else:
                #### 将seq_num步的sgcn_outs，拼接起来，shape：seq_num*N*hidden_dims_gcn[-1]
                sgcn_outs = torch.cat((sgcn_outs, cur), dim=0)          
        lstm_outs = self.lstm(sgcn_outs)            #### lstm_out shape: seq_num*N*lstm_output_dims or seq_num*N*hidden_dims_lstm
        lstm_outs = lstm_outs.view(self.seq_num, -1)
        outs = self.fc1(lstm_outs)
        outs = self.fc2(outs)
        outs = self.fc3(outs)
        return outs




def norm(adjacency):
    adjacency += np.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    degree = np.diag(np.power(degree, -0.5).flatten())
    return degree.dot(adjacency).dot(degree)

#### 最大最小值归一化
def NormData(data):
    tmp = np.max(data, axis=0) - np.min(data, axis=0)
    for i in range(len(tmp)):
        if tmp[i]==0:
            tmp[i] = 1
    data = (data-np.min(data, axis=0))/tmp
    return data

def train(traindata, adjacencies, Targets, net, criterion, optimizer, epoch):     #### traindata: (seq_sum+1)*N*d;  adjacencies: seq_sum*N*N
    for i in range(epoch):
        running_loss = 0
        for j in range(traindata.shape[0]-net.seq_num):
            start = j
            end = j+net.seq_num
            inputs_x = traindata[start:end, :, :]
            inputs_adj = adjacencies[start:end, :]
            # targets = traindata[start+1:end+1, :, :]
            targets = Targets[start+1:end+1, :]

            optimizer.zero_grad()

            outs = net(inputs_x, inputs_adj)
            loss = criterion(outs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch %d , running loss = %.4f" % (i, running_loss))

    print("Finish Training!")



def test(testdata, adjacencies, Targets,  net, criterion, optimizer):

    pred = []
    ground = []
    with torch.no_grad():
        for i in range(testdata.shape[0] - net.seq_num):
            test_loss = 0

            start = i
            end = i + net.seq_num
            inputs_x = testdata[start:end,:,:]
            inputs_adj = adjacencies[start:end, :, :]
            targets = Targets[start+1:end+1,:]
            # print("inputs_x size: ", inputs_x.size())
            # print("inputs_adj size: ", inputs_adj.size())

            predictions = net(inputs_x, inputs_adj)
            # pred.append(predictions[])
            loss = criterion(predictions, targets)
            test_loss += loss.item()

            pred.append(predictions[-1])
            ground.append(targets[-1])

            print("Seq %d , test loss = %.4f " % (i, test_loss))
    print('Finishing testing!')
    ViewResult(pred, ground)
     

def RandData(seq_num, N, d):
    traindata = torch.randn(seq_num, N, d)
    adjacencies = np.random.random(seq_num*N*N)
    for i in range(len(adjacencies)):
        if adjacencies[i]<0.8:
            adjacencies[i] = 0
    adjacencies = np.reshape(adjacencies, (seq_num, N, N))
    for i in range(seq_num):
        adjacencies[i] = norm(adjacencies[i])
    adjacencies = torch.from_numpy(adjacencies)
    print("traindatain", traindata.size())
    print("adjacencies", adjacencies.size())
    return traindata, adjacencies

def PrepareData(filepath, seq_sum, AdjSize, col_start, col_end):
    data = [[] for _ in range(seq_sum)]
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        cnt = 0
        cur_seq = -1
        for row in f_csv:
            if cnt%AdjSize == 0:
                cur_seq += 1
            if cur_seq>=seq_sum:
                break
            one_item = []
            for v in range(col_start, col_end):
                one_item.append(float(row[v]))
            data[cur_seq].append(one_item)
            cnt += 1
    data = np.asarray(data)
    data = np.reshape(data, (-1, (col_end-col_start)))
    data = NormData(data)
    data = np.reshape(data, (seq_sum, AdjSize, (col_end-col_start)))
    data = torch.Tensor(data)
    # print("traindata size: ", data.size())
    return data

def PrepareTarget(filepath_target, seq_sum, batch_size, output_dims):
    Target = []
    with open(filepath_target, 'r') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)

        cnt = 0
        for row in f_csv:
            if cnt>seq_sum:
                break 
            cur_data = [float(row[i]) for i in range(1, output_dims+1)]
            Target.append(cur_data)
    Target = np.asarray(Target)
    Target = NormData(Target)
    Target = torch.tensor(Target)
    return Target




def PrepareAdj(filepath, seq_sum, AdjSize):
    adjacencies = []
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        
        for row in f_csv:
            cur_adj = [float(v) for v in row]
            adjacencies.append(cur_adj)
    # adjacencies = torch.tensor(adjacencies)
    # adjacencies = adjacencies.view(-1,AdjSize, AdjSize)
    adjacencies = np.asarray(adjacencies)
    adjacencies = np.reshape(adjacencies, (-1, AdjSize, AdjSize))
    # for i in range(adjacencies.shape[0]):
    #     adjacencies[i] = norm(adjacencies[i])
    adjacencies = torch.tensor(adjacencies)
    # adjacencies = adjacencies.view(-1,AdjSize, AdjSize)
    # print("adjacencies size: ", adjacencies.size())
    return adjacencies


def ViewResult(pred, ground ):
    length = len(pred)
    x = range(length)
    feature_num = pred[0].size(0)
    print("feature_num: ", feature_num)

    headers = ['nodes','layer_0', 'layer_1','layer_2','layer_3','layer_4','layer_5']
    # print("pred: ", pred[0][0])
    # print("groud: ", ground[0][0])

    for i in range(feature_num):
        pred_v = []
        ground_v = []
        for j in range(length):
            pred_v.append(pred[j][i].item())
            ground_v.append(ground[j][i].item())

        plt.plot(x, pred_v, label = "pred")
        plt.plot(x, ground_v, label = 'groud')
        plt.title(headers[i])
        plt.legend()
        plt.show()



     

if __name__ == '__main__':
    seq_num, input_dims, output_dims= 6, 11, 5
    hidden_dims_lstm = 4
    hidden_dims_gcn = [2, 16, 8]
    batch_size, layers_lstm  = 1000, 1
    AdjSize = batch_size
    seq_sum = 90            #### 指多少个月的数据
    # net = LstmGcnNet(seq_num, input_dims, hidden_dims, output_dims, batch_size, layers)
    net = LstmGcnNet(seq_num, input_dims, output_dims, hidden_dims_lstm, hidden_dims_gcn, batch_size, layers_lstm)
    learning_rate, momentum = 0.05, 0.9
    epoch = 200
    start_train, end_train = 20, 70
    start_test, end_test = end_train-seq_num+1, 90

    print(net)

    # # traindata, adjacencies = RandData(31+10, batch_size, input_dims)
    FileRootpath = 'GetDataCode\\data\\'
    filepath_data = FileRootpath + "Net_Data_32755.csv"
    filepath_adj = FileRootpath + "Net_Adj_32755.csv"
    filepath_target = FileRootpath + "Net_32755_LayerData.csv"
    Data = PrepareData(filepath_data, seq_sum, batch_size, 4, 4+input_dims)
    print("Data shape: ", Data.size())
    TargetData = PrepareTarget(filepath_target, seq_sum, batch_size, output_dims)
    print("TargetData shape: ", TargetData.size())
    # print(Data[0][0])
    Adjacencies = PrepareAdj(filepath_adj, seq_sum, batch_size)
    print("Adjanceies shape: ", Adjacencies.size())

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    train(Data[start_train:end_train,:,:], Adjacencies[start_train:end_train,:,:], TargetData[start_train:end_train, :], net, criterion, optimizer, epoch)
    test(Data[start_test:end_test, :, :], Adjacencies[start_test:end_test, :, :], TargetData[start_test:end_test, :], net, criterion, optimizer)

