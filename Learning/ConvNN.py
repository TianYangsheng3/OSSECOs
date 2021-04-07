import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flot_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def num_flot_features(self, x):
        size = x.size()[1:]
        print("size: ", size)
        num_feature = 1
        for s in size:
            num_feature *= s
        # print("num_feature: ", num_feature)
        return num_feature

net = Net()
print(net)

params = list(net.parameters())
# print("params: ", params)
print("params len : ",len(params))
for i in range(len(params)):
    print("params layer_"+str(i)+" size: ", params[i].size())

input = torch.randn(1,1,32,32)
print("input: ", input.size())
out  = net(input)
print("out: ", out)

# net.zero_grad()
# out.backward(torch.randn(1,10))

target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()
# loss = criterion(out, target)
# print("loss: ", loss)


# net.zero_grad()
# print(net.conv1.bias.requires_grad)
# net.conv1.bias.retain_grad()
# print("conv1.bias.grad before backward:", net.conv1.bias.grad)
# loss.backward()
# print("conv1.bias.grad after backward:", net.conv1.bias.grad)
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    print("conv1.bias.grad time_"+str(i)+": ", net.conv1.bias.grad)
    optimizer.step()




