import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
import csv
import matplotlib.pyplot as plt

class Lstm(nn.Module):

    def __init__(self, input_dims, hidden_dims, layers, batch_size):
        super().__init__()
        self.layers = layers
        self.input_dims = input_dims
        self.hidden_dims  = hidden_dims
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dims, hidden_dims, layers)
        # self.fc1 = nn.Linear(hidden_dims, 10)
        # self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        h0 = torch.randn(self.layers, self.batch_size, self.hidden_dims)
        c0 = torch.randn(self.layers, self.batch_size, self.hidden_dims)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        return out

def pre_data(filepath, seq_num, batch, features):
    data = []
    # np.seterr(divide='ignore', invalid='ignore')
    sourcedata = np.zeros((seq_num, batch, features))
    count = 0
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append([float(i) for i in row])
            count += 1
            if count==320:
                break
    data = np.array(data)
    tmp = np.max(data, axis=0)- np.min(data, axis=0)
    for tmp_i in range(len(tmp)):
        if tmp[tmp_i] == 0:
            tmp[tmp_i] = 1
    data = (data - np.min(data, axis=0))/tmp

    for row in range(len(data)):
        for i in range(seq_num):
                for j in range(features):
                    cur = i*features + j
                    # print(cur)
                    sourcedata[i][row][j] = data[row][cur]
    print(sourcedata.shape)


    return sourcedata

def train(traindata, lstm, criterion, optimizer, batch_size, seq_num, epoch): ### data 50*320*23
    # seq_num = 50
    # epoch = 50

    # input_dims, hidden_dims, layers, batch_size = 22, 23, 2, 8
    # lstm = Lstm(input_dims, hidden_dims, layers, batch_size)
    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(lstm.parameters(), lr=0.1, momentum=0.9)
    
    for i in range(epoch):
        for batch_i in range(int(traindata.size(1)/batch_size)):
            running_loss = 0

            start = batch_i*batch_size
            end = (batch_i+1)*batch_size
            inputs = traindata[:seq_num, start:end, 1:]
            target = traindata[1:seq_num+1, start:end, :]

            optimizer.zero_grad()

            outputs = lstm(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_i == (int(traindata.size(1)/batch_size)-1):
                print('[%d, %5d] loss: %.3f ' % (i+1, batch_i+1, running_loss))

            if batch_i == (int(traindata.size(1)/batch_size)-1) and i==epoch-1:
                cmp(i, target, outputs)

    print("Finishing Training")


def test(testdata, lstm, criterion, optimizer, batch_size, seq_num):
    # test_loss = 0
    with torch.no_grad():
        for batch_i  in range(int(testdata.size(1)/batch_size)):
            test_loss = 0
            start = batch_i*batch_size
            end = (batch_i+1)*batch_size
            inputs = testdata[:seq_num, start:end, 1:]
            target = testdata[1:seq_num+1, start:end, :]

            predictions = lstm(inputs)
            loss = criterion(predictions, target)
            test_loss += loss.item()
            print('testdata: [%5d] loss: %.3f ' % (batch_i+1, test_loss))
            if batch_i == (int(testdata.size(1)/batch_size)-1) :
                cmp(0, target, predictions)
    
    print("Finishing testing!")


def cmp(epoch, groud, pred):
    batch_size = groud.size(1)
    seq_num = groud.size(0)
    features = groud.size(2)
    x = range(seq_num)
    for batch_i in range(batch_size):

        for features_i in range(features):
            groud_y = []
            pred_y = []
            for j in range(seq_num):
                groud_y.append(groud[j][batch_i][features_i].item())
                pred_y.append(pred[j][batch_i][features_i].item())
            plt.plot(x, groud_y, label = "Groud")
            plt.plot(x, pred_y, label = "pred")
            plt.legend()
            plt.title("epoch " + str(epoch) + " batch " + str(batch_i) + " feature : " + str(features_i))
            filepath = "fig2//"+"epoch_"+str(epoch) + "_batch_" + str(batch_i) + "_feature_" + str(features_i)+".pdf"
            plt.savefig(filepath, bbox_inches = 'tight')
            plt.close()


    




if __name__ == '__main__':
    filepath = "file//month_lstm.csv"
    data = pre_data(filepath, 51, 320, 23)
    data = torch.from_numpy(data)
    data = data.to(torch.float32)
    # print(data)
    rate = 0.75
    traindata, testdata = data[:, :int(320*rate), :], data[:, int(320*rate): , :]
    print("traindata, testdata : ", traindata.size(), testdata.size())
    seq_num = 50
    epoch = 100
    input_dims, hidden_dims, layers, batch_size = 22, 23, 2, 8
    lstm = Lstm(input_dims, hidden_dims, layers, batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(lstm.parameters(), lr=0.1, momentum=0.9)

    #print("before training: ", lstm.lstm._all_weights[0])
    train(traindata, lstm, criterion, optimizer, batch_size, seq_num, epoch)
    # print("after training: ", lstm.lstm.bias)
    test(testdata, lstm, criterion, optimizer, batch_size, seq_num)
