#!/usr/bin/env python
import math
from time import sleep

import numpy as np

import rospy
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray
from NeurovisInterface import NeurovisInterface
from Nvis3D import NeuroVis3D
from threading import Thread

ninterface = NeurovisInterface()
N3d = None

def setNames(data):
    try:
        ninterface.createNeuronsFromNames(data.data)
    except Exception as e:
        print("setNames")
        print(e.__str__())
        pass

def setPositions(data):
    while(True):
        if ( ninterface.populated):
            try:
                ninterface.setNeuronLayersIn3D(data.data)
            except Exception as e:
                print("setPositions")
                print(e.__str__())
                pass
            break

        else:
            sleep(0.1)

def setConnections(data):
    size = int(math.sqrt(len(data.data)))
    arr = np.array(data.data)
    arr = np.resize(arr,(size,size))
    while(True):
        if ( ninterface.populated):
            try:
                ninterface.setConnectionsWithConnArr(arr)
            except Exception as e:
                print("setConnections")
                print(e.__str__())
            break
        else:
            sleep(0.1)

def fullyConnectLayers(data):

    while(True):
        if ( ninterface.populated):
            try:
                #L1-L2/L2-L3/ etc

                layers = data.data.split("/")
                for i in layers:
                    if i:
                        splitted = i.split("-")
                        fromL = splitted[0]
                        toL = splitted[1]

                    ninterface.setConnectionsWithLayers(fromL,toL)
            except Exception as e:
                print("fullyConnectLayers")
                print(e.__str__())
            break
        else:
            sleep(0.1)

def setDisplayText(data):

    while(True):
        if ( ninterface.populated):
            try:
                ninterface.setDisplayText(data.data)
            except Exception as e:
                print("setDisplayText")
                print(e.__str__())
            break
        else:
            sleep(0.1)

def setActivity(data):
    arr = np.array(data.data)
    while(True):
        if ( ninterface.populated):
            try:
                ninterface.setActivity(arr)
            except Exception as e:
                print("setActivity")

                print(e.__str__())
            break
        else:
            sleep(0.1)

def createDisplay(data):
    arr = np.array(data.data)

    while(True):
        if ( ninterface.populated):
            try:
                id = arr[0]
                size = (arr[1], arr[2])
                nvis.gc.addToDisplayTab(id,size)
            except Exception as e:
                print("createDisplay")

                print(e.__str__())
            break
        else:
            sleep(0.1)

def updateDisplay(data):
    arr = np.array(data.data)

    while(True):
        if ( ninterface.populated):
            try:
                id = arr[0]
                imgArr = arr[1:]
                nvis.gc.updateDisplayTab(id,imgArr)
            except Exception as e:
                print("updateDisplay")
                print(e.__str__())
            break
        else:
            sleep(0.1)

def listener():
    rospy.get_caller_id()
    rospy.Subscriber("/neurovis/neuronName", String, setNames)
    rospy.Subscriber("/neurovis/neuronPos", String, setPositions)
    rospy.Subscriber("/neurovis/connections", Float32MultiArray, setConnections)
    rospy.Subscriber("/neurovis/connectionsLayers", String, fullyConnectLayers)

    rospy.Subscriber("/neurovis/activity", Float32MultiArray, setActivity)

    rospy.Subscriber("/neurovis/createDisplay", Int32MultiArray, createDisplay)
    rospy.Subscriber("/neurovis/updateDisplay", Int32MultiArray, updateDisplay)

    rospy.Subscriber("/neurovis/setDisplayText", String, setDisplayText)



    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

rospy.init_node('NeuroVis3D')
rate = rospy.Rate(10)

nThread = Thread(target=listener)
nThread.start()

nvis = NeuroVis3D(ninterface)
nvis.start()
nvis.run()






