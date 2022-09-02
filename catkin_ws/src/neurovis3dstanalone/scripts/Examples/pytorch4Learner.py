import random

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
Y = torch.Tensor([[0], [1], [1], [1]])

class NET2(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(NET2, self).__init__()

        self.activity = np.zeros(5)
        self.weights = np.zeros(6)


        self.in1 = nn.Linear(1, 1, bias=False)
        self.in2 = nn.Linear(1, 1, bias=False)
        self.in1.weight.data = torch.tensor([[1.0]])
        self.in2.weight.data = torch.tensor([[1.0]])
        self.h1 = nn.Linear(2, 1, bias=True)
        self.h2 = nn.Linear(2, 1, bias=True)

        self.h3 = nn.Linear(2, 1, bias=True)



    def forward(self, x):
        A1 = self.in1(x[0].view(1))
        B1 = self.in2(x[1].view(1))

        self.activity[0] = A1
        self.activity[1] = B1

        A2 = self.h1(torch.cat((A1, B1), 0))
        A2 = F.sigmoid(A2)

        B2 = self.h2(torch.cat((A1, B1), 0))
        B2 = F.sigmoid(B2)

        self.activity[2] = A2
        self.activity[3] = B2

        x = self.h3(torch.cat((A2, B2)))
        AB3 = F.sigmoid(x)
        self.activity[4] = AB3


        self.weights = np.array([self.h1.weight.data[0][0],self.h1.weight.data[0][1],self.h2.weight.data[0][0],self.h2.weight.data[0][1],self.h3.weight.data[0][0],self.h3.weight.data[0][1]])


        return AB3


model = NET2()

# some tests
print(model(torch.tensor([1.0, 0.0])))
print(model(torch.tensor([0.0, 1.0])))
print(model(torch.tensor([1.0, 1.0])))
print(model(torch.tensor([0.0, 0.0])))


# connection arr by hand cus its faster
connArr = np.array([
    [0.0, 0.0, 1, -1, 0.0],
    [0.0, 0.0, 1, -1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1],
    [0.0, 0.0, 0.0, 0.0, 1],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])
connArr = connArr.flatten()


# name_layer_x_y


# name_layer_x_y
pos = "I-1_0_0_0/I-2_0_2_0/A-1_1_0_0/B-1_1_2_0/AB_2_1_0/"
names = "I-1/I-2/A-1/B-1/AB/"


# now for the ros part


rospy.init_node('pytorch2', anonymous=True)

pubName = rospy.Publisher('/neurovis/neuronName', String, queue_size=1)
pubCon = rospy.Publisher('/neurovis/connections', Float32MultiArray, queue_size=1)
pubPos = rospy.Publisher('/neurovis/neuronPos', String, queue_size=1)
pubAct = rospy.Publisher('/neurovis/activity', Float32MultiArray, queue_size=1)
pubWe = rospy.Publisher('/neurovis/weightsAll', Float32MultiArray, queue_size=1)


pubCreaDis = rospy.Publisher('/neurovis/createDisplay', Int32MultiArray, queue_size=1)
pubUpdDis = rospy.Publisher('/neurovis/updateDisplay', Int32MultiArray, queue_size=1)
pubText = rospy.Publisher("/neurovis/setDisplayText", String, queue_size=1)



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


rate = rospy.Rate(1)  # 3hz

while pubName.get_num_connections() <= 0:
    rate.sleep()

pubName.publish(names)

while pubCon.get_num_connections() <= 0:
    rate.sleep()

publish1DArr(pubCon, connArr)
pubPos.publish(pos)


#publish1DArr(pubWeight, updatedWeights)
publish1DArr(pubAct, model.activity)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


while not rospy.is_shutdown():
    data_point = X[np.random.randint(X.size(0))]
    #we train the thing
    input = 0
    output = 0
    outputs = None
    for i in range(20):
        ran = random.randint(0,len(X)-1)
        input, output = X[ran],Y[ran]
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, output)
        loss.backward()
        optimizer.step()
    publish1DArr(pubAct, model.activity)
    publish1DArr(pubWe, model.weights)
    #print(model.weights)
    pubText.publish("Err: " + str(outputs - output))

    rate.sleep()
