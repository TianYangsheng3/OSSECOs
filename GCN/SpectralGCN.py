import numpy as np
import torch,csv
# torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from collections import namedtuple 
import matplotlib.pyplot as plt

class GraphConvolution(nn.Module):

    def __init__(self, input_dims, output_dims, use_bias = True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dims, output_dims))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dims))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, adjacency, inputs):           #### 邻接矩阵：adjacency，shape: N*N ;  图节点的特征矩阵 inputs， shape：N*d,其中d=input_dims
        support = torch.mm(inputs, self.weight)     #### weight shape: d*output_dims
        output = torch.sparse.mm(adjacency, support)            #### output shape: N*output_dims
        if self.use_bias:
            output += self.bias
        return output

class LSTM(nn.Module):
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

# class GcnNet(nn.Module):

#     def __init__(self, input_dims = 1433):
#         super().__init__()
#         self.gcn1 = GraphConvolution(input_dims, 16)
#         self.gcn2 = GraphConvolution(16, 7)

#     def forward(self, adjacency, inputs):
#         h = F.relu(self.gcn1(adjacency, inputs))
#         outputs = self.gcn2(adjacency, h)
#         return outputs

class LstmGcnNet(nn.Module):
    def __init__(self, seq_num, input_dims, hidden_dims, output_dims, batch_size, layers):
        super().__init__()
        self.seq_num = seq_num
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.layers = layers
        self.Gcn = GraphConvolution(input_dims, output_dims)
        self.lstm = LSTM(input_dims, output_dims, batch_size, layers)
    
    def forward(self, adjacencies, xs):        #### adjacencies shape: seq*N*N ; xs shape: seq*N*d, d = input_dims, N = batch_size
        # print("forward adjacencies: ", adjacencies.size())
        for i in range(self.seq_num):
            cur = F.relu(self.Gcn(adjacencies[i], xs[i]))          #### cur shape: N*output_dims
            cur = cur.view(-1, self.batch_size, self.output_dims)
            if i==0:
                gcn_output = cur 
            else:
                gcn_output = torch.cat((gcn_output, cur), dim=0)
        # print("gcn_output: ", gcn_output.size())
        lstm_out = self.lstm(gcn_output)                #### lstm_out shape: seq*N*output_dims
        return lstm_out


def norm(adjacency):
    adjacency += np.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    degree = np.diag(np.power(degree, -0.5).flatten())
    return degree.dot(adjacency).dot(degree)


def train(traindata, adjacencies, net, criterion, optimizer, epoch):     #### traindata: (seq_sum+1)*N*d;  adjacencies: seq_sum*N*N
    for i in range(epoch):
        running_loss = 0
        for j in range(traindata.shape[0]-net.seq_num):
            start = j
            end = j+net.seq_num
            inputs_x = traindata[start:end, :, :]
            inputs_adj = adjacencies[start:end, :]
            targets = traindata[start+1:end+1, :, :]

            optimizer.zero_grad()

            outs = net(inputs_adj, inputs_x)
            loss = criterion(outs[-1:,:,:], targets[-1:,:,:])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch %d , running loss = %.4f" % (i, running_loss))

    print("Finish Training!")



def test(testdata, adjacencies, net, criterion, optimizer):

    pred = []
    ground = []
    with torch.no_grad():
        for i in range(testdata.shape[0] - net.seq_num):
            test_loss = 0

            start = i
            end = i + net.seq_num
            inputs_x = testdata[start:end,:,:]
            inputs_adj = adjacencies[start:end, :, :]
            targets = testdata[start+1:end+1,:,:]
            # print("inputs_x size: ", inputs_x.size())
            # print("inputs_adj size: ", inputs_adj.size())

            predictions = net(inputs_adj, inputs_x)
            # pred.append(predictions[])
            loss = criterion(predictions[-1:,:,:], targets[-1:,:,:])
            test_loss += loss.item()

            pred.append(predictions[-1:,:,:])
            ground.append(targets[-1:,:,:])

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
            one_item = []
            for v in range(col_start, col_end):
                one_item.append(float(row[v]))
            data[cur_seq].append(one_item)
            cnt += 1
    data = torch.Tensor(data)
    # print("traindata size: ", data.size())
    return data

def PrepareAdj(filepath, seq_sum, AdjSize):
    adjacencies = []
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        
        for row in f_csv:
            cur_adj = [float(v) for v in row]
            adjacencies.append(cur_adj)
    adjacencies = torch.tensor(adjacencies)
    adjacencies = adjacencies.view(-1,AdjSize, AdjSize)
    # print("adjacencies size: ", adjacencies.size())
    return adjacencies


def ViewResult(pred, ground ):
    length = len(pred)
    x = range(length)
    node_num = pred[0].size(1)
    feature_num = pred[0].size(2)

    pred_degree = []
    ground_degree = []
    for i in range(length):
        cur_pred = pred[i].view(node_num, feature_num)
        # print(torch.sum(cur_pred, dim=0)[-2])
        pred_degree.append(torch.sum(cur_pred, dim=0)[-1].item())
        cur_ground = ground[i].view(node_num, feature_num)
        ground_degree.append(torch.sum(cur_ground, dim=0)[-1].item())
    plt.plot(x, pred_degree, label = "pred")
    plt.plot(x, ground_degree, label = 'groud')
    plt.legend()
    plt.show()



     

if __name__ == '__main__':
    seq_num, input_dims, hidden_dims, output_dims, batch_size, layers = 5, 13, 13, 13, 500, 2
    AdjSize = 500
    net = LstmGcnNet(seq_num, input_dims, hidden_dims, output_dims, batch_size, layers)
    
    learning_rate, momentum = 0.1, 0.9
    epoch = 500
    # print(net)

    # traindata, adjacencies = RandData(31+10, batch_size, input_dims)
    FileRootpath = 'GetDataCode\\data\\'
    filepath_data = FileRootpath + "Net_Data_4023.csv"
    filepath_adj = FileRootpath + "Net_Adj_4023.csv"
    Data = PrepareData(filepath_data, 73, batch_size, 4, 17)
    Adjacencies = PrepareAdj(filepath_adj, 73, batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    train(Data[0:31,:,:], Adjacencies[0:31,:,:], net, criterion, optimizer, epoch)
    test(Data[27:51, :, :], Adjacencies[27:51, :, :],  net, criterion, optimizer)
