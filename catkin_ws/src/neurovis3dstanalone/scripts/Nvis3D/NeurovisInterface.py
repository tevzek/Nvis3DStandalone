import math
import random
from Panda3dUtils import moveNeuronAndItsControlPointsRelativeToOther
import numpy as np

import re

from numpy import isnan

from Neuron import Neuron, NeuronConnection, NeuronLayer


class NeurovisInterface:


  def __init__(self):
    self.correlationMethod = ""
    self.corelate = False
    self.neurons = {}
    self.inputNeurons = {}
    self.layers = {}
    self.setLayerInfo = False

    self.neuronIds = 0
    self.conIds = 0
    self.populated = False

    #corelation variables
    self.step = 0.1
    self.corCutoff = 0.8
    self.corArraySize = 10
    self.globalCorrelation = False
    
    self.maxPullingDistance = 5
    self.maxPushingingDistance = 30
    self.amIPushing = True
    self.hardPushingWallDistance = 5

    self.amIsetingDisplayText = False
    self.textToSet = ""

  def createNeuronsFromNames(self, namesStr):
    splited = namesStr.split("/")
    for i in splited:
      if i:
        if i not in self.neurons.keys():
          self.neurons[i] = Neuron(i,np.array([1,1,1]),1)

    self.populated = True

    pass

  def setDisplayText(self,text):
    self.amIsetingDisplayText = True
    self.textToSet = text

  def findIndexOfNeuronById(self,id):
    #method finds neuron by id todo make with pointers cus speed
    for x in range(len(self.neurons)):
        if self.neurons[x].id == id:
            return x
    else:
        return -1

  def getNeuronByName(self, name):
    return self.neurons[name]

  def connectLayers(self,A,B):
    l1 = self.layers[A]
    l2 = self.layers[B]

  #todo put in another class?
  def reasignBezierControllPoints(self,con):

    #method that reasigns controll points
    fn = con.fromNeuron
    tn = con.toNeuron
    if fn == None or tn == None:
      print("Cannot change bezier control point")
      return
    if((fn.pos[0] == tn.pos[0]) and (fn.pos[1]==tn.pos[1]) and (fn.pos[2] == tn.pos[2])):
      con.controlPoint = [fn.pos[0] + random.randrange(30,60),fn.pos[1] + random.randrange(30,60),fn.pos[2] + random.randrange(30,60)]
      con.originalpControlPointPos = con.controlPoint.copy()

    else:
      middleground = [(fn.pos[0]+tn.pos[0])/2,(fn.pos[1]+tn.pos[1])/2,(fn.pos[2]+tn.pos[2])/2]
      middleground[0]+= float(random.randrange(-10,10))
      middleground[1]+= float(random.randrange(-10,10))
      middleground[2]+= float(random.randrange(-10,10))
      con.controlPoint = middleground
      con.originalpControlPointPos = middleground.copy()

  def clearConns(self):
    for n in range(len(self.neurons)):
      self.neurons[n].connectionsOut = []

  def getConnectionActivity(self, connect):
    return connect.fromNeuron.activity

  def setNeuronLayersIn3D(self,posStr):

    zStep = 6
    yStep = 5
    xStep = 5
    nc = self.neurons.copy()
    self.inputNeurons = {}
    posSplit = posStr.split("/")
    for i in posSplit:
      split = i.split("_")
      #we split it in name, Layer, x, y
      if not split or len(split) != 4:
        print("Neuron isn't named for 3d view continuing without the layering display")
        continue
      try :
        self.neurons[split[0]].pos[0] = float(split[2]) * xStep * 5
        self.neurons[split[0]].pos[1] = float(split[3]) * yStep * 5
        self.neurons[split[0]].pos[2] = float(split[1]) * zStep * 5


        self.neurons[split[0]].originalpPos[0] = float(split[2]) * xStep * 5
        self.neurons[split[0]].originalpPos[1] = float(split[3]) * yStep * 5
        self.neurons[split[0]].originalpPos[2] = float(split[1]) * zStep * 5

        layerName = split[0].split("-")[0]
        if layerName not in self.layers:
          self.layers[layerName] = NeuronLayer(layerName)


        self.layers[layerName].addNeuron(self.neurons[split[0]])
        self.neurons[split[0]].layer = layerName

        x = re.search("^I", split[0])
        if x:
          self.inputNeurons[split[0]] = self.neurons[split[0]]
      except Exception as e:
        print(e)
        self.neurons = nc
        print("Neurons aren't named for 3d view continuing without the layering display")

        continue
    self.createNewBezierControlPoints()
    self.setLayerInfo = True

  def setNeuronsToNeuroVis(self):
    #copy own neurons to neurovis
    self.__nvis.set_neurons(self.neurons)

  def createNewBezierControlPoints(self):
    keys = list(self.neurons.keys())
    for k in keys:
      for c in self.neurons[k].connectionsOut:
        fromN = c.fromNeuron
        toN = c.toNeuron
        center = (fromN.pos + toN.pos)
        for i in range(len(center)):
          center[i] += random.uniform(-5.0, 5.0)
        c.controlPoint = center

  def createNewBezierPoint(self,conn):
        fromN = conn.fromNeuron
        toN = conn.toNeuron
        center = (fromN.pos + toN.pos)
        for i in center:
          i += random.uniform(-5.0, 5.0)

  def setConnectionsWithConnArr(self,conArr):
    keys = list(self.neurons.keys())

    for k,n in zip(keys,range(len(keys))):
      for c, indix in zip(conArr[n],range(len(conArr[n]))):
        if c != 0:
          fromN = self.neurons[k]
          toN = self.neurons[keys[indix]]

          conn = NeuronConnection(fromN,toN,weight=c)
          self.conIds +=1
          fromN.connectionsOut.append(conn)
          toN.connectionsIn.append(conn)
          self.createNewBezierPoint(conn)

  def setConnectionsWithLayers(self, fromL,toL):

    #fully connect two layers

    fr = self.layers[fromL].neurons
    to = self.layers[toL].neurons

    for n1 in fr.values():
      for n2 in to.values():
        fromN = n1
        toN = n2
        conn = NeuronConnection(fromN, toN)
        fromN.connectionsOut.append(conn)
        toN.connectionsIn.append(conn)
        self.createNewBezierPoint(conn)

  def setActivity(self,act):
    for n,a in zip(list(self.neurons.items()),act):
      n[1].activity = a
      n[1].previusActivityArr = np.roll(n[1].previusActivityArr,-1)
      n[1].previusActivityArr[-1] = a
    #every time we get a input we move the neurons
    if(self.corelate):
      self.corelateMovment()

  def setWeights(self,arr):
    count = int(len(arr)//3)
    indx = 1
    while indx<=count:
      fromm = arr[indx*count]
      to = arr[indx * count + 1]
      weight = arr[indx * count + 2]
      indx += 1
      listt = list(self.neurons.items())
      fromN = listt[fromm]
      toN = listt[to]
      conn = self.neurons[fromN].findConnectionOutByName(toN)
      conn.updateWeight(weight)

  def setWeightsAll(self,arr):
    indx = 0
    for n in list(self.neurons.values()):
      for c in n.connectionsOut:
        c.updateWeight(arr[indx])
        indx += 1

  def stopCorelationMovment(self):
    self.correlationMethod = ""
    self.corelate = False

  def startCorelationMovment(self,method):
    self.correlationMethod = method
    self.corelate = True

  def moveNeuronsByCorelation(self,n1,n2,cor):
    maxstr = 1 - self.corCutoff
    
    hardWallDist = self.maxPullingDistance
    if self.amIPushing:
      # times 2 to transform it to 2d cords, and +1 cus there seems to be a rounding err
      hardWallDist = self.hardPushingWallDistance


    fn = n1  # from neuron is the neuron we are connected to so yea
    tn = n2
    moveVec = tn.pos-fn.pos

    # vectorLenght
    vecLen = np.linalg.norm(moveVec)

    moveVec = self.normalize(moveVec)

    corVal = cor
    # if we attract it or repulse it
    # attract
    if corVal >= self.corCutoff:
      if vecLen > self.maxPullingDistance and vecLen > hardWallDist:
        tmp = corVal - self.corCutoff
        str = tmp / maxstr
        moveVec = moveVec * str * self.step
      else:
        return
    else:
      if vecLen < self.maxPushingingDistance:
        str = corVal / maxstr
        str = 1 - str
        moveVec = moveVec * (-str) * self.step
      else:
        return
          # we move the vec
    newPos = n1.pos + moveVec/2
    moveNeuronAndItsControlPointsRelativeToOther(n1, newPos)
    newPos = n2.pos - moveVec/2
    moveNeuronAndItsControlPointsRelativeToOther(n2, newPos)

    return

  def corelateMovment(self):
    if self.correlationMethod == "pearson":
      for k,n in list(self.neurons.items()):
        #by default we only corelate to neurons that we are connected to
        if self.globalCorrelation == False:
          for c in n.connectionsOut:
            r2 = np.corrcoef(c.fromNeuron.previusActivityArr, c.toNeuron.previusActivityArr)[0][1]
            if isnan(r2):
              r2 = 1
            r2 = math.fabs(r2)
            self.moveNeuronsByCorelation(c.fromNeuron,c.toNeuron,r2)
          for c in n.connectionsIn:
            r2 = np.corrcoef(c.toNeuron.previusActivityArr, c.fromNeuron.previusActivityArr)[0][1]
            if isnan(r2):
              r2 = 1
            r2 = math.fabs(r2)
            self.moveNeuronsByCorelation(c.toNeuron,c.fromNeuron,r2)
          #else we check every neuron
        else:
          for k, n in list(self.neurons.items()):
            for k2, n2 in list(self.neurons.items()):
              if k != k2:
                r2 = np.corrcoef(n.previusActivityArr, n2.previusActivityArr)[0][1]
                if isnan(r2):
                  r2 = 1
                r2 = math.fabs(r2)
                self.moveNeuronsByCorelation(n, n2, r2)

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm == 0:
      return v
    return v / norm



