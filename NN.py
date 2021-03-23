import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import random
import numpy as np
import numpy.random as rd
import csv

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        # self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        

        return x



def data(num, feature):
    all_data = []

    def fun(feature, cur):  
        res = 0
        bias = np.random.normal(0, 0.01)
        for i in range(feature):
            res = res + (i+1)*cur[i]*0.1
        return res+bias
    
    for j in range(num):
        cur = []
        for k in range(feature):
            tmp = random.random()
            cur.append(tmp)
        res = fun(feature, cur)
        cur.append(res)
        all_data.append(cur)
    
    with open('file/data.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(all_data)

    return all_data


if __name__ == '__main__':
    net = Net()
    net = net.double()
    print("net: ", net)
    params = list(net.parameters())
    # print("params: ", params)
    print("params len : ",len(params))

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    Data_ = np.array(data(100, 3))
    Data = Data_/Data_.max(axis=0)
    # print(np.shape(Data))
    inputd = torch.from_numpy(Data[:,0:3])
    # print(inputd)
    target = torch.from_numpy(Data[:, 3:])
    # print(target)

    for i in range(10000):
        optimizer.zero_grad()
        pred = net(inputd)
        loss = loss_fn(pred, target)
        # if i==9:
        #     print("pred: ", pred)

        # print("time "+str(i)+" loss: ", loss)
        loss.backward()
        optimizer.step()
        if i==9999:
            pred = pred.detach().numpy()
            target = target.detach().numpy()
            with open('file/pred.csv', 'w') as f:
                f_csv = csv.writer(f)
                length = len(pred)
                for k in range(length):
                    f_csv.writerow([target[k][0], pred[k][0]])