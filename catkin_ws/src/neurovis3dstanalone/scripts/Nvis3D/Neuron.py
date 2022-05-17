import math
import random

import numpy as np


class Neuron:
  idd = 0
  tmp = 0
  def __init__(self,name,pos = np.array([0,0,0]),activity = 0.0):
    self.name = name


    self.pos = pos.copy()
    self.pos = np.array(self.pos)

    self.activity = activity
    self.id = Neuron.idd
    Neuron.idd += 1
    self.connectionsOut = []
    self.connectionsIn = []
    self.model = None

    self.previusActivityArrSize = 10
    self.previusActivityArr = np.zeros(10)

    self.originalpPos = self.pos.copy()
    self.layer = ""
  def get3DPos(self):
    try:
      #return ([self.pos[0]/10,self.pos[1]/10,self.pos[2]/10])
      return (self.pos)
    except:
      print("Error getting 3d pos of neuron "+self.name)
      return (np.array([0,0,0]))

  def set3DPos(self, pos):
    self.pos = np.array(pos)

class NeuronConnection:
  idd = 0
  def __init__(self,fromN,toN,weight=0.5,controlPoint=np.zeros(3),arrowPos=np.zeros(3),previousWeight=0):
    self.weight = weight

    self.controlPoint = controlPoint.copy()

    self.arrowPosition = arrowPos
    self.previousWeight = previousWeight

    self.model = None
    self.fromNeuron = fromN
    self.toNeuron = toN

    self.id= NeuronConnection.idd
    NeuronConnection.idd += 1

    self.originalpControlPointPos = self.controlPoint.copy()
    self.correlationToOutNeurons = []




    pass


  def get3DControlPoint(self):

    return [self.controlPoint[0],self.controlPoint[1], self.controlPoint[2]]
class NeuronLayer:
  def __init__(self,name):
    #neuron layer
    self.name = name
    self.neurons = {}
    self.minMaxX = [100000,-100000]
    self.minMaxY = [100000,-100000]
    self.minMaxZ = [100000,-100000]
    self.simpleModel = None
  def addNeuron(self,n):
    #we add a neuron and update cords
    margin = 2.0
    if n.name not in self.neurons:
      self.neurons[n.name] = n
      if n.pos[0] < self.minMaxX[0]:
        self.minMaxX[0] = n.pos[0] - margin
      if n.pos[0] > self.minMaxX[1]:
        self.minMaxX[1] = n.pos[0] + margin

      if n.pos[1] < self.minMaxY[0]:
        self.minMaxY[0] = n.pos[1] - margin
      if n.pos[1] > self.minMaxY[1]:
        self.minMaxY[1] = n.pos[1] + margin

      if n.pos[2] < self.minMaxZ[0]:
        self.minMaxZ[0] = n.pos[2] - margin
      if n.pos[2] > self.minMaxZ[1]:
        self.minMaxZ[1] = n.pos[2] + margin
  def updateBoundsByNeuron(self):
    #we add a neuron and update cords
    margin = 2.0
    self.minMaxX = [100000,-100000]
    self.minMaxY = [100000,-100000]
    self.minMaxZ = [100000,-100000]
    for n in list(self.neurons.values()):
      if n.name in self.neurons:
        if n.pos[0] < self.minMaxX[0]:
          self.minMaxX[0] = n.pos[0] - margin
        if n.pos[0] > self.minMaxX[1]:
          self.minMaxX[1] = n.pos[0] + margin

        if n.pos[1] < self.minMaxY[0]:
          self.minMaxY[0] = n.pos[1] - margin
        if n.pos[1] > self.minMaxY[1]:
          self.minMaxY[1] = n.pos[1] + margin

        if n.pos[2] < self.minMaxZ[0]:
          self.minMaxZ[0] = n.pos[2] - margin
        if n.pos[2] > self.minMaxZ[1]:
          self.minMaxZ[1] = n.pos[2] + margin
    #and now we reshape the layer
    layerToWorkOn = self
    if layerToWorkOn.simpleModel is not None:
      center = [(layerToWorkOn.minMaxX[0] + layerToWorkOn.minMaxX[1]) / 2,
                (layerToWorkOn.minMaxY[0] + layerToWorkOn.minMaxY[1]) / 2,
                (layerToWorkOn.minMaxZ[0] + layerToWorkOn.minMaxZ[1]) / 2]
      sqSize = 3.2808398950131235
      roof = self.simpleModel
      roof.setPos(center[0], center[1], center[2])
      roof.setColor(1,1,1, 1)
      roof.setP(-90)

      lenX = math.fabs(layerToWorkOn.minMaxX[0] - layerToWorkOn.minMaxX[1])
      roof.setSx(lenX / sqSize)

      lenY = math.fabs(layerToWorkOn.minMaxY[0] - layerToWorkOn.minMaxY[1])
      roof.setSz(lenY / sqSize)

