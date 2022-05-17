import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, Int32MultiArray

import rospy

torch.manual_seed(2)

X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)


class NET1(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(NET1, self).__init__()

        self.activity = np.zeros(5)

        self.in1 = nn.Linear(1, 1, bias=False)
        self.in2 = nn.Linear(1, 1, bias=False)
        self.h1 = nn.Linear(1, 1, bias=False)
        self.h2 = nn.Linear(1, 1, bias=False)
        self.out = nn.Linear(2, output_dim, bias=False)

        self.in1.weight.data = torch.tensor([[1.0]])
        self.in2.weight.data = torch.tensor([[1.0]])
        # self.in1.bias.data = torch.tensor([0.0])
        # self.in2.bias.data = torch.tensor([0.0])

        self.h1.weight.data = torch.tensor([[1.0]])
        self.h2.weight.data = torch.tensor([[-1.0]])
        self.out.weight.data = torch.tensor([[1.0, -1.0]])

    def forward(self, x):
        A1 = self.in1(x[0].view(1))
        B1 = self.in2(x[1].view(1))

        self.activity[0] = A1
        self.activity[1] = B1

        A2 = self.h1(A1)
        A2 = F.sigmoid(A2)

        B2 = self.h2(B1)
        B2 = F.sigmoid(B2)

        self.activity[2] = A2
        self.activity[3] = B2

        x = self.out(torch.cat((A2, B2)))
        O = F.sigmoid(x)
        self.activity[4] = O

        return O


model = NET1()

# some tests
print(model(torch.tensor([1.0, 0.0])))
print(model(torch.tensor([0.0, 1.0])))
print(model(torch.tensor([1.0, 1.0])))
print(model(torch.tensor([0.0, 0.0])))

# connection arr by hand cus its faster
connArr = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])
connArr = connArr.flatten()

updatedWeights = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

# name_layer_x_y


# name_layer_x_y
pos = "I-1_0_0_0/I-2_0_2_0/A-1_1_0_0/B-1_1_2_0/O_2_1_0/"
names = "I-1/I-2/A-1/B-1/O/"

# now for the ros part


rospy.init_node('pytorch1', anonymous=True)

pubName = rospy.Publisher('/neurovis/neuronName', String, queue_size=1)
pubCon = rospy.Publisher('/neurovis/connections', Float32MultiArray, queue_size=1)
pubPos = rospy.Publisher('/neurovis/neuronPos', String, queue_size=1)
pubAct = rospy.Publisher('/neurovis/activity', Float32MultiArray, queue_size=1)

pubCreaDis = rospy.Publisher('/neurovis/createDisplay', Int32MultiArray, queue_size=1)
pubUpdDis = rospy.Publisher('/neurovis/updateDisplay', Int32MultiArray, queue_size=1)


# a.layout.dim.append(thing)
def publish1DArr(pub, arr):
    a = Float32MultiArray()
    a.data = arr
    dim = MultiArrayDimension()
    dim.label = "x"
    dim.size = len(arr)
    dim.stride = len(arr)
    a.layout.dim.append(dim)
    a.layout.data_offset = 0
    pub.publish(a)
def publish1DArrInt(pub, arr):
    a = Int32MultiArray()
    a.data = arr
    dim = MultiArrayDimension()
    dim.label = "x"
    dim.size = len(arr)
    dim.stride = len(arr)
    a.layout.dim.append(dim)
    a.layout.data_offset = 0
    pub.publish(a)

rate = rospy.Rate(10)  # 3hz


while pubName.get_num_connections() <= 0:
    rate.sleep()

pubName.publish(names)

while pubCon.get_num_connections() <= 0:
    rate.sleep()

publish1DArr(pubCon, connArr)
print(connArr)
pubPos.publish(pos)
publish1DArrInt(pubCreaDis,[1,2,3])

while not rospy.is_shutdown():
    data_point = X[np.random.randint(X.size(0))]
    model(torch.tensor(data_point))
    publish1DArr(pubAct, model.activity)
    arr = np.concatenate((model.activity, np.array([model.activity[-1]])))
    arr *= 255
    zer = np.zeros(arr.shape[0]*3)
    zer[1::3] = arr
    zer = np.concatenate((np.array([1]), zer))
    publish1DArrInt(pubUpdDis, zer.astype(np.int32))

    rate.sleep()
