import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from os.path import exists
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension, Int16MultiArray, Int16, Int32MultiArray

import rospy



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.fc1Arr = np.zeros(16 * 5 * 5)
        self.fc2Arr = np.zeros(120)
        self.fc3Arr = np.zeros(84)
        self.OutArr = np.zeros(10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        self.fc1Arr = x.detach().numpy().flatten()
        x = F.relu(self.fc1(x))
        self.fc2Arr = x.detach().numpy().flatten()
        x = F.relu(self.fc2(x))
        self.fc3Arr = x.detach().numpy().flatten()
        x = self.fc3(x)
        self.OutArr = x.detach().numpy().flatten()
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

PATH = './cifar_net.pth'
file_exists = exists(PATH)
if not file_exists:
    print("didn't find a save file for the NN")
    print("Started traning")
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

#now we do ros stuff
pos = ""
names = ""

for y in range(5):
    for x in range(16):
        for z in range(5):
            pos += "C1-{0}-{1}-{2}_{3}_{1}_{2}/".format(y,x,z,y*0.2)
            names += "C1-{0}-{1}-{2}/".format(y, x,z)
for y in range(8):
    for x in range(15):
            pos += "C2-{0}-{1}_2_{0}_{1}/".format(y,x)
            names += "C2-{0}-{1}/".format(y,x)

for y in range(7):
    for x in range(12):
            pos += "C3-{0}-{1}_3_{0}_{1}/".format(y,x)
            names += "C3-{0}-{1}/".format(y,x)
for y in range(10):
    pos += "O-{0}_4_{0}_0/".format(y)
    names += "O-{0}/".format(y)

rospy.init_node('classifier', anonymous=True)

pubName = rospy.Publisher('/neurovis/neuronName', String, queue_size=1)
pubCon = rospy.Publisher('/neurovis/connectionsLayers', String, queue_size=1)
pubPos = rospy.Publisher('/neurovis/neuronPos', String, queue_size=1)
pubAct = rospy.Publisher('/neurovis/activity', Float32MultiArray, queue_size=1)

pubCreaDis = rospy.Publisher('/neurovis/createDisplay', Int32MultiArray, queue_size=1)
pubUpdDis = rospy.Publisher('/neurovis/updateDisplay', Int32MultiArray, queue_size=1)

pubText = rospy.Publisher("/neurovis/setDisplayText", String, queue_size=1)


rate = rospy.Rate(0.3) #0.3hz


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

while pubPos.get_num_connections() <= 0:
    rate.sleep()

pubPos.publish(pos)

while pubCon.get_num_connections() <= 0:
    rate.sleep()
#pubCon.publish("C1-C2/C2-C3/C3-O/")

correct = 0
total = 0

publish1DArrInt(pubCreaDis,[1,64,64])




while not rospy.is_shutdown():
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network

            outputs = net(images)
            numpyarr = np.zeros((64, 64, 3))
            #unmormilize

            images = torchvision.utils.make_grid(images,padding=0)

            images = images / 2 + 0.5
            numpyImages = images.numpy()
            numpyImages = np.transpose(numpyImages, (1, 2, 0))

            numpyarr[0:32, 0:32, ::] = numpyImages[0:32,0:32,::]
            numpyarr[32:64, 0:32, ::] = numpyImages[0:32,32:64,::]
            numpyarr[0:32, 32:64, ::] = numpyImages[0:32,64:96:, ::]
            numpyarr[32:64, 32:64, ::] = numpyImages[0:32,96:128:, ::]
            numpyarr = np.rot90(numpyarr,2)

            numpyarr *= 255

            numpyarr = numpyarr.flatten()
            numpyarr = np.concatenate(([1],numpyarr)).astype(np.int32)
            numpyarr = np.clip(numpyarr, 0, 255)
            publish1DArrInt(pubUpdDis, numpyarr)

            #publish the act.
            act = np.concatenate((net.fc1Arr, net.fc2Arr, net.fc3Arr,net.OutArr))
            publish1DArr(pubAct, act)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            outText = ""
            for i in predicted:
                outText+= " " + classes[i]
            pubText.publish(outText)

            rate.sleep()



print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')