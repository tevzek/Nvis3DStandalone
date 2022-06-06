import math
import cv2

import numpy
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, Int16MultiArray, Int16, Int32MultiArray
from torchvision import transforms
import rospy
import numpy as np
from PIL import Image

picSize = 200
power = 1.0
dec = True


torch.manual_seed(2)

finalNnSize = int(((((picSize-4)/2)-4)/2/2))**2

fakeFrame = Image.open("./blueBall.jpg")


class NET3(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(NET3, self).__init__()
        #self.conv1 = nn.Conv2d(1, 3, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.relu = nn.Linear(finalNnSize,1,bias=False)
        self.out = nn.Linear(1, 1,bias=False)

        self.reluOutArr = np.zeros((23*23))
        self.relu1OutArr = np.zeros((1))
        self.outOutArr = np.zeros((1))


    def forward(self, x):
        #(1*200*200)
        #x = self.conv1(x)
        #I'll just reduce the size so I don't have to re work it
        x = x[::,::,4::,4::]
        x = self.pool(x)
        #(3*196*196)
        #the filter size makes it smaller for 5 its 5-1 so 4
        #(3*98*98)
        x = self.conv2(x)
        # (1*94*94)
        # the filter size makes it smaller for 5 its 5-1 so 4
        # (1*47*47)
        x = self.pool(x)
        x = self.pool(x)
        # (1*23*23)

        x = torch.flatten(x, 1)
        a = x.detach().numpy()


        self.reluOutArr = a.copy().flatten()

        self.relu.weight.data = x * 0.2

        # flatten all dimensions except batch
        x = F.relu(self.relu(x))
        a = x.detach().numpy()
        self.relu1OutArr = a.copy().flatten()



        x = F.tanh(self.out(x))
        a = x.detach().numpy()
        self.outOutArr = a.copy().flatten()

        return x



vc = cv2.VideoCapture(2)


def getNextFrame():
    global power
    global dec


    frame = None
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        frame = np.zeros((3,3,3))


    imageIn = cv2.resize(frame, dsize=(picSize, picSize), interpolation=cv2.INTER_CUBIC)
    #image = fakeFrame.resize((picSize,picSize),Image.ANTIALIAS)
    #image = convert_tensor(imageIn)
    #image *= power
    image = np.copy(imageIn)
    image = image.astype(int)

    truArr = ((image[::, ::, 0] + image[::, ::, 1]) * 0.7) < image[::, ::, 2]
    if dec:
        power -= 0.1
        if power <= 0:
            dec = False
    else:
        power += 0.1
        if power >= 1:
            dec = True

    image = image[::,::,2]
    image = np.where(truArr, image, 0)
    return (image,imageIn)
net = NET3()




tens, _ = getNextFrame()
tens = tens / 255
tens = torch.from_numpy(tens)
tens = torch.reshape(tens, (1,1,picSize, picSize))
tens = tens.float()
print(net(tens))
tt = 0

conArr = np.zeros((finalNnSize+2,finalNnSize+2))
conArr[0:finalNnSize:,-2] = 1
conArr[-2,-1] = 1

connArr = conArr.flatten()

pos = ""
names = ""
siz = int(math.sqrt(finalNnSize))

for y in range(siz):
    for x in range(siz):
        pos += "C-{0}-{1}_0_{2}_{3}/".format(y,x,y,x)
        names += "C-{0}-{1}/".format(y, x)
pos += "T_1_0_0/"
pos += "O_2_0_0/"
names += "T/O/"


rospy.init_node('pytorch3', anonymous=True)

pubName = rospy.Publisher('/neurovis/neuronName', String, queue_size=1)
pubCon = rospy.Publisher('/neurovis/connections', Float32MultiArray, queue_size=1)
pubPos = rospy.Publisher('/neurovis/neuronPos', String, queue_size=1)
pubAct = rospy.Publisher('/neurovis/activity', Float32MultiArray, queue_size=1)

pubCreaDis = rospy.Publisher('/neurovis/createDisplay', Int32MultiArray, queue_size=2)
pubUpdDis = rospy.Publisher('/neurovis/updateDisplay', Int32MultiArray, queue_size=1)

fwd = rospy.Publisher('/morf_hcmd/morf_fwd_speed', Float32MultiArray, queue_size=1)


rate = rospy.Rate(3)  # 3hz


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

while pubName.get_num_connections() <= 0:
    rate.sleep()

pubName.publish(names)

while pubCon.get_num_connections() <= 0:
    rate.sleep()

publish1DArr(pubCon, connArr)
pubPos.publish(pos)

pubPos.publish(pos)
publish1DArrInt(pubCreaDis,[1,picSize,picSize])
publish1DArrInt(pubCreaDis,[2,picSize,picSize])


indx = 0



while not rospy.is_shutdown():
    tens, org = getNextFrame()

    tens = tens / 255
    tens = torch.from_numpy(tens)
    tens = torch.reshape(tens, (1, 1, picSize, picSize))
    tens = tens.float()
    out = net(tens)

    tempPic = tens.detach().numpy()
    tempPic = np.abs(tempPic)
    tempPic = np.rot90(tempPic, 1)
    tempPic *= 255
    tempPic = tempPic.flatten()
    tempPic = np.clip(tempPic, 0, 255)
    tempEmpyPic = np.zeros( tempPic.shape[0]*3)
    tempEmpyPic[1::3] = tempPic
    tempPic = tempEmpyPic
    tempPic = np.concatenate(([1],tempPic)).astype(np.int32)

    publish1DArrInt(pubUpdDis, tempPic)

    tempPic = org
    tempPic = tempPic.flatten()
    tempPic = np.clip(tempPic, 0, 255)
    tempPic = np.concatenate(([2],tempPic)).astype(np.int32)

    publish1DArrInt(pubUpdDis, tempPic)

    act = np.concatenate((net.reluOutArr,net.relu1OutArr,net.outOutArr))
    publish1DArr(pubAct, act)

    publish1DArr(fwd,net.relu1OutArr)

    rate.sleep()


