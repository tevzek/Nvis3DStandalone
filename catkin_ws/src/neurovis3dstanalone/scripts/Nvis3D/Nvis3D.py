import random
import time

import bezier
import numpy
import numpy as np
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import LPoint3
import math

from direct.gui.DirectGui import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import AmbientLight, DirectionalLight, LineSegs, NodePath, CollisionTraverser, CollisionNode, \
    TextNode, \
    CollisionSphere, CollisionHandlerPusher, Texture, GeomVertexFormat, Geom, GeomVertexData, GeomVertexWriter, Vec3, \
    GeomTriangles, GeomNode
from panda3d.core import Vec4
from panda3d.core import WindowProperties
from Panda3dUtils import KeyControler, guiControler
import Panda3dUtils


class NeuroVis3D(ShowBase):

    def getColorFromValues(self, val, maxVal=1.0, minVal=-1.0):
        valc = val
        if valc > maxVal:
            valc = maxVal
        if valc < minVal:
            valc = minVal

        dist = math.fabs(maxVal) + math.fabs(minVal)
        normalisedVal = math.fabs(math.fabs(valc - minVal))
        power = normalisedVal / dist
        pg = 0
        pr = 0
        if power > 0.5:
            pg = (power - 0.5) * 2
        if power < 0.5:
            pr = math.fabs((0.5 - power) * 2)

        b = 0.2
        return (pr, pg, b)

    def makeSphere(self, pos=(5, 0, 0), col=(1, 0, 0), name="N"):

        NeuronNode = self.render.attachNewNode(name)

        sphere = self.loader.loadModel("res/PandaRes/Sphere_HighPoly.egg")
        sphere.reparentTo(NeuronNode)
        NeuronNode.setPos(pos[0], pos[1], pos[2])
        sphere.setColor(col[0], col[1], col[2], 1)

        cNode = CollisionNode('colNode')
        cNode.addSolid(CollisionSphere(0, 0, 0, 3))

        sphereC = sphere.attachNewNode(cNode)
        sphereC.hide()
        tt = 0
        # Set the object's collision node to render as visible.

        # Add the Pusher collision handler to the collision traverser.
        #self.picker.addCollider(sphereC, self.pusher)
        # Add the 'frowney' collision node to the Pusher collision handler.
        #self.pusher.addCollider(sphereC, sphere, self.drive.node())

        return NeuronNode

    def angle_between(self, v1, v2):

        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * (180 / math.pi)

    def updateNeuronsByEachNeuron(self):
        for key,i in self.nvisInterface.neurons.copy().items():
            ncol = self.getColorFromValues(i.activity)
            # if we dont have this neuron we create it
            if i.model == None:
                self.drawANeuron(i)
            else:
                sphere = i.model.getChild(0)
                if self.clickedNeuron == None or self.clickedNeuron.id != i.id:
                    sphere.setColor(ncol[0], ncol[1], ncol[2], 1)
                poss = i.get3DPos()

                # i.model.posInterval(0.1, Point3(poss[0], poss[1], poss[2]), startPos=i.model.getPos(), fluid=1)
                i.model.setPos(poss[0], poss[1], poss[2])

            for j in i.connectionsOut:
                '''
                connectedId = self.nvisInterface.findIndexOfNeuronById(j.connIds[1])
                connectedN = self.nvisInterface.neurons[connectedId]
                # cols = self.getColorFromValues(j.weight)
                act = self.nvisInterface.getConnectionActivity(i, j)
                # todo ask zumo if ths is ok
                '''
                cols = self.getColorFromValues(i.activity)
                if j.model == None:
                    # self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=2.5, col = cols)
                    # l = self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=3.5, col = cols)
                    l = self.makeLineBezier(j.fromNeuron.get3DPos(), j.toNeuron.get3DPos(),
                                            controlPoint=j.get3DControlPoint(), width=2.5, col=cols, arrow=True)

                    j.model = l
                else:
                    self.updateBezierLine(j)
                    line = j.model[0]
                    # line.setColor(cols[0],cols[1],cols[2],1)
                    vert = line.getVertices()

                    for v in range(len(vert)):
                        line.setVertexColor(v, cols[0], cols[1], cols[2])
                    cone = j.model[1]
                    if cone is not None:
                        cone.setColor(cols[0], cols[1], cols[2],1)


                    tt = 0

                # we make a arrow here

    def updateNeuronsByEachLayer(self):
        for lkey,li in self.nvisInterface.layers.copy().items():

            if lkey not in self.layersToSimplify:
                if li.simpleModel is not None:
                    if not li.simpleModel.isHidden():
                        li.simpleModel.hide()

                #same code as in updateNeuronsByEachNeuron
                for key,i in li.neurons.copy().items():
                    ncol = self.getColorFromValues(i.activity)
                    # if we dont have this neuron we create it
                    if i.model == None:
                        self.drawANeuron(i)
                    else:
                        #unhide it
                        if i.model.isHidden():
                            i.model.show()
                        sphere = i.model.getChild(0)
                        if self.clickedNeuron == None or self.clickedNeuron.id != i.id:
                            sphere.setColor(ncol[0], ncol[1], ncol[2], 1)
                        poss = i.get3DPos()

                        # i.model.posInterval(0.1, Point3(poss[0], poss[1], poss[2]), startPos=i.model.getPos(), fluid=1)
                        i.model.setPos(poss[0], poss[1], poss[2])

                    for j in i.connectionsOut:
                        '''
                        connectedId = self.nvisInterface.findIndexOfNeuronById(j.connIds[1])
                        connectedN = self.nvisInterface.neurons[connectedId]
                        # cols = self.getColorFromValues(j.weight)
                        act = self.nvisInterface.getConnectionActivity(i, j)
                        # todo ask zumo if ths is ok
                        '''
                        cols = self.getColorFromValues(i.activity)
                        if j.model == None:
                            # self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=2.5, col = cols)
                            # l = self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=3.5, col = cols)
                            l = self.makeLineBezier(j.fromNeuron.get3DPos(), j.toNeuron.get3DPos(),
                                                    controlPoint=j.get3DControlPoint(), width=2.5, col=cols, arrow=True)

                            j.model = l
                        else:
                            self.updateBezierLine(j)
                            line = j.model[0]
                            # line.setColor(cols[0],cols[1],cols[2],1)
                            vert = line.getVertices()
                            for v in range(len(vert)):
                                line.setVertexColor(v, cols[0], cols[1], cols[2])

                            cone = j.model[1]
                            if cone is not None:
                                cone.setColor(cols[0], cols[1], cols[2], 1)

                            tt = 0

                        # we make a arrow here
            else:
                if li.simpleModel == None:
                    # we draw a box around the layer here and hide the neurons
                    #we draw the layer if it dosent exsist
                    layerToWorkOn = li
                    center = [(layerToWorkOn.minMaxX[0] + layerToWorkOn.minMaxX[1])/2,(layerToWorkOn.minMaxY[0] + layerToWorkOn.minMaxY[1])/2,(layerToWorkOn.minMaxZ[0] + layerToWorkOn.minMaxZ[1])/2]
                    sqSize = 3.2808398950131235
                    roof = self.loader.loadModel("res/PandaRes/Square.egg")
                    roof.reparentTo(self.render)
                    roof.setPos(center[0], center[1], center[2])
                    roof.setColor(1, 1, 1, 1)
                    roof.setP(-90)

                    lenX = math.fabs(layerToWorkOn.minMaxX[0] - layerToWorkOn.minMaxX[1])
                    roof.setSx(lenX/sqSize)

                    lenY = math.fabs(layerToWorkOn.minMaxY[0] - layerToWorkOn.minMaxY[1])
                    roof.setSz(lenY / sqSize)

                    #add texture
                    myTexture = Texture("LayerTexture {0}".format(layerToWorkOn.name))
                    # texture size
                    myTexture.setup2dTexture(len(li.neurons), 1, Texture.T_unsigned_byte, Texture.F_rgb)
                    myTexture.setMagfilter(Texture.FT_nearest)
                    roof.setTexture(myTexture)

                    li.simpleModel = roof

                #show if hidden
                elif li.simpleModel.isHidden():
                    li.simpleModel.show()
                if not list(li.neurons.values())[0].model.isHidden():
                    for n in list(li.neurons.values()):
                        n.model.hide()

                textureActList = []
                for i in list(li.neurons.values()):
                    #update the texture
                    textureActList.append(0)
                    textureActList.append(i.activity)
                    textureActList.append(0)

                    # we still have to update the conns
                    for j in i.connectionsOut:
                        '''
                        connectedId = self.nvisInterface.findIndexOfNeuronById(j.connIds[1])
                        connectedN = self.nvisInterface.neurons[connectedId]
                        # cols = self.getColorFromValues(j.weight)
                        act = self.nvisInterface.getConnectionActivity(i, j)
                        # todo ask zumo if ths is ok
                        '''
                        cols = self.getColorFromValues(i.activity)
                        if j.model == None:
                            # self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=2.5, col = cols)
                            # l = self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=3.5, col = cols)
                            l = self.makeLineBezier(j.fromNeuron.get3DPos(), j.toNeuron.get3DPos(),
                                                    controlPoint=j.get3DControlPoint(), width=2.5, col=cols, arrow=True)

                            j.model = l
                        else:
                            self.updateBezierLine(j)
                            line = j.model[0]
                            # line.setColor(cols[0],cols[1],cols[2],1)
                            vert = line.getVertices()
                            for v in range(len(vert)):
                                line.setVertexColor(v, cols[0], cols[1], cols[2])
                            cone = j.model[1]
                            if cone is not None:
                                cone.setColor(cols[0], cols[1], cols[2], 1)

                #swap textures
                textureActList = np.array(textureActList)
                textureActList *= 255
                textureActList = textureActList.astype(np.uint8).T
                textureActList = textureActList.tostring()
                tex = li.simpleModel.getTexture()
                tex.setRamImage(textureActList)
                li.simpleModel.setTexture(tex)

    def updateNeurons(self, task):

        if not self.nvisInterface.layers:
            self.updateNeuronsByEachNeuron()
        else:
            self.updateNeuronsByEachLayer()
        return Task.cont

    def drawANeuron(self, neuronN):
        i = neuronN
        ncol = self.getColorFromValues(i.activity)
        n = self.makeSphere(i.get3DPos(), ncol, name=i.name)
        i.model = n
        for j in i.connectionsOut:
            connectedN = j.toNeuron
            # cols = self.getColorFromValues(j.weight)
            act = self.nvisInterface.getConnectionActivity( j)
            # todo ask zumo if ths is ok
            cols = self.getColorFromValues(act)
            # self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=2.5, col = cols)
            # l = self.makeLine(i.get3Dpos(), connectedN.get3Dpos(), width=2.5, col = cols)
            l = self.makeLineBezier(i.get3DPos(), connectedN.get3DPos(), controlPoint=j.get3DControlPoint(),
                                    width=2.5, col=cols, arrow=True)
            j.model = l

        # we draw text on top of neurons here
        text = TextNode('test textNode')
        text.setText(i.name)
        text3d = NodePath(text)
        text3d.setScale(2, 2, 2)
        text3d.setTwoSided(True)

        tex3dTop = text3d.__copy__()
        tex3dBottom = text3d.__copy__()

        tex3dFront = text3d.__copy__()
        tex3dBack = text3d.__copy__()

        tex3dRight = text3d.__copy__()
        tex3dLeft = text3d.__copy__()

        tex3dTop.reparentTo(n)
        tex3dTop.setPos(-1, 0, 3.5)
        tex3dTop.setHpr(0, -90, 0)

        tex3dBottom.reparentTo(n)
        tex3dBottom.setPos(-1, 0, -3.5)
        tex3dBottom.setHpr(0, -90, 0)

        tex3dFront.reparentTo(n)
        tex3dFront.setPos(-1, -3.5, 0)

        tex3dBack.reparentTo(n)
        tex3dBack.setPos(-1, 3.5, 0)

        tex3dRight.reparentTo(n)
        tex3dRight.setHpr(90, 0, 0)
        tex3dRight.setPos(3.5, 0, 0)

        tex3dLeft.reparentTo(n)
        tex3dLeft.setHpr(90, 0, 0)
        tex3dLeft.setPos(-3.5, 0, 0)

    def drawNeurons(self):
        #self.nvisInterface.updateNeuronsGracefull()
        tmp = self.nvisInterface.neurons.copy()
        for key,i in tmp.items():
            self.drawANeuron(i)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def moveNeuronsByCorelation(self, task, step, corCutoff):
        maxstr = 1 - corCutoff
        hardWallDist = self.maxPullingDistance
        if self.amIPushing:
            # times 2 to transform it to 2d cords, and +1 cus there seems to be a rounding err
            hardWallDist = self.hardPushingWallDistance*2 +1

        for n in self.nvisInterface.neurons:
            for c in n.connectionsIn:
                tn = c.fromNeuron  # from neuron is the neuron we are connected to so yea
                fn = c.toNeuron
                moveVec = np.array([tn.pos[0] - fn.pos[0], tn.pos[1] - fn.pos[1], tn.pos[2] - fn.pos[2]])

                # vectorLenght
                vecLen = np.linalg.norm(moveVec)

                moveVec = self.normalize(moveVec)


                if len(c.correlationToOutNeurons) > 0:
                    corVal = c.correlationToOutNeurons.pop(0)
                    # if we attract it or repulse it
                    #attract
                    if corVal >= corCutoff:

                        if vecLen > self.maxPullingDistance and vecLen > hardWallDist:
                            tmp = corVal - corCutoff
                            str = tmp / maxstr
                            moveVec = moveVec * str * step
                        else:
                            return Task.cont

                    else:
                        if vecLen < self.maxPushingingDistance:
                            str = corVal / corCutoff
                            str = 1 - str
                            moveVec = moveVec * (-str) * step
                        else:
                            return Task.cont
                    # we move the vec
                    newPos = [moveVec[0] + n.pos[0], moveVec[1] + n.pos[1], moveVec[2] + n.pos[2]]
                    Panda3dUtils.moveNeuronAndItsControlPointsRelativeToOther(n, newPos)
            #copy paste code for out connection only swap from to
            for c in n.connectionsOut:
                tn = c.toNeuron  # from neuron is the neuron we are connected to so yea
                fn = c.fromNeuron
                moveVec = np.array([tn.pos[0] - fn.pos[0], tn.pos[1] - fn.pos[1], tn.pos[2] - fn.pos[2]])

                # vectorLenght
                vecLen = np.linalg.norm(moveVec)

                moveVec = self.normalize(moveVec)


                if len(c.correlationToOutNeurons) > 0:
                    corVal = c.correlationToOutNeurons.pop(0)
                    # if we attract it or repulse it
                    #attract
                    if corVal >= corCutoff:

                        if vecLen > self.maxPullingDistance and vecLen > hardWallDist:
                            tmp = corVal - corCutoff
                            str = tmp / maxstr
                            moveVec = moveVec * str * step
                        else:
                            return Task.cont

                    else:
                        if vecLen < self.maxPushingingDistance:
                            str = corVal / corCutoff
                            str = 1 - str
                            moveVec = moveVec * (-str) * step
                        else:
                            return Task.cont
                    # we move the vec
                    newPos = [moveVec[0] + n.pos[0], moveVec[1] + n.pos[1], moveVec[2] + n.pos[2]]
                    Panda3dUtils.moveNeuronAndItsControlPointsRelativeToOther(n, newPos)


        return Task.cont

    def pushNeuronsOutOfProximity(self, task):
        # here we calculate distance from one neuron to others and push them so they dont touch and repulse them away
        # it's a O(n²) so we might think of something smarter, maybe panda3d colliders
        # I made it n * logn
        if not self.amIPushing:
            return Task.cont


        nsize = len(self.nvisInterface.neurons)

        hardWall = self.hardPushingWallDistance
        softWall = self.softPushingWallDistance
        softWallStr = self.softPushingWallStrenght
        keys = list(self.nvisInterface.neurons.keys())
        for nA in range(nsize):
            n1 = self.nvisInterface.neurons[keys[nA]]
            for nB in range(nA + 1, nsize):
                n2 = self.nvisInterface.neurons[keys[nB]]

                posN1 = n1.pos
                posN2 = n2.pos

                #get the vector from first to second neuron
                vec = posN2-posN1
                dist = np.linalg.norm(vec)

                #if the distance is within our proxyimity we push the other neuron away
                if dist < hardWall:

                    norV = self.normalize(vec)
                    pushingV = norV * hardWall
                    newPos = posN1 + pushingV
                    n2.set3DPos(newPos)

                    #we need ti calculate the dist again
                    posN1 = n1.get3DPos()
                    posN2 = n2.get3DPos()
                    vec = posN2-posN1
                    dist = np.linalg.norm(vec)

                if dist < softWall:
                    #we push the two neurons in each others area
                    norV = self.normalize(vec)
                    pushingV = norV * softWallStr
                    newPos = posN2 + pushingV
                    n2.set3DPos(newPos)

                    pushingV*=-1
                    newPos = posN1 + pushingV
                    n1.set3DPos(newPos)

        #display the area on selected neuron


        return Task.cont

    def startCorelationMovment(self, step, corCutoff, arrSize, method):

        self.nvisInterface.step = step
        self.nvisInterface.corCutoff = corCutoff
        self.nvisInterface.corArraySize = arrSize


        # this starts colecting corelation data
        self.nvisInterface.startCorelationMovment(method)

    def stopCorelationMovment(self):
        self.nvisInterface.stopCorelationMovment()

    def makeLine(self, fromm=[0, 0, 0], to=[2, 2, 0], width=2.0, col=(0.8, 0.2, 0.2)):


        lines = LineSegs()
        lines.setColor(col[0], col[1], col[2], 1)
        lines.moveTo(fromm[0], fromm[1], fromm[2])
        lines.drawTo(to[0], to[1], to[2])
        lines.setThickness(width)
        node = lines.create()
        np = NodePath(node)
        return lines

    def makeLineBezier(self, fromm=[0, 0, 0], to=[2, 2, 0], controlPoint=[3, 1, 0], width=2.0, points=20,
                       col=(0.8, 0.2, 0.2), arrow=True):


        node1 = [fromm[0], controlPoint[0], to[0]]
        node2 = [fromm[1], controlPoint[1], to[1]]
        node3 = [fromm[2], controlPoint[2], to[2]]
        nodes = [node1, node2, node3]

        curve = bezier.Curve(nodes, degree=2)
        curvePoints = curve.evaluate_multi(numpy.linspace(0.0, 1.0, points))

        lines = LineSegs()
        lines.setColor(col[0], col[1], col[2], 1)
        lines.moveTo(fromm[0], fromm[1], fromm[2])

        for i in range(len(curvePoints[1])):
            lines.drawTo(curvePoints[0][i], curvePoints[1][i], curvePoints[2][i])
            # lines.moveTo(curvePoints[0][i],curvePoints[1][i],curvePoints[2][i])
        lines.setThickness(width)
        node = lines.create("lines")
        ndp = NodePath(node)
        ndp.reparentTo(self.render)


        if arrow:
            arrowPos = np.array([curvePoints[0][points//2],curvePoints[1][points//2],curvePoints[2][points//2]])
            next1 = np.array([curvePoints[0][points//2+1],curvePoints[1][points//2+1],curvePoints[2][points//2+1]])

            cone = self.loader.loadModel("res/PandaRes/Cone.egg")
            cone.setPos(arrowPos[0], arrowPos[1], arrowPos[2])
            cone.setColor(0.5, 0.2, 0.2, 1)
            cone.reparentTo(self.render)
            cone.getChild(0).setP(-90)

            cone.lookAt(next1[0],next1[1],next1[2])

            return (lines, cone)

        return (lines, None)
    def drawBorders(self, task):
        #we draw the borders here
        if self.clickedNeuron != None and self.showPushingDistance:
            pos = self.clickedNeuron.get3DPos()
            self.softWallSphere.show()
            self.hardWallSphere.show()
            self.softWallSphere.setPos(pos[0],pos[1],pos[2])
            self.hardWallSphere.setPos(pos[0],pos[1],pos[2])
            self.softWallSphere.setScale((1/(self.sphereDim*2))*self.softPushingWallDistance*2)
            self.hardWallSphere.setScale((1/(self.sphereDim*2))*self.hardPushingWallDistance*2)
        else:
            self.softWallSphere.hide()
            self.hardWallSphere.hide()


        return task.cont

    def updateBezierLine(self, conn, points=20):
        fromm = conn.fromNeuron.get3DPos()
        try:
            to = conn.toNeuron.get3DPos()
        except:
            to = conn.toNeuron.get3DPos()

        controlPoint = conn.controlPoint

        node1 = [fromm[0], controlPoint[0], to[0]]
        node2 = [fromm[1], controlPoint[1], to[1]]
        node3 = [fromm[2], controlPoint[2], to[2]]
        nodes = [node1, node2, node3]

        curve = bezier.Curve(nodes, degree=2)
        curvePoints = curve.evaluate_multi(numpy.linspace(0.0, 1.0, points))

        # 0 is the model of line 1 is the arrow
        line = conn.model[0]
        lineSegmentsNum = line.getNumVertices() - 1

        line.setVertex(0, curvePoints[0][0], curvePoints[1][0], curvePoints[2][0])

        for i, j in zip(range(lineSegmentsNum), range(len(curvePoints[0]))):
            line.setVertex(i + 1, curvePoints[0][j], curvePoints[1][j], curvePoints[2][j])
        cone = conn.model[1]
        if cone is not None:
            arrowPos = np.array([curvePoints[0][points//2],curvePoints[1][points//2],curvePoints[2][points//2]])
            next = np.array([curvePoints[0][points//2+1],curvePoints[1][points//2+1],curvePoints[2][points//2+1]])
            cone.setPos(arrowPos[0],arrowPos[1],arrowPos[2])
            cone.lookAt(next[0],next[1],next[2])


        tt = 0

    def makeCone(self, pos=[2, 2, 0], col=[0.8, 0.5, 0.2], rot=(0, [0, 0, 0])):

        pivotNode = self.render.attachNewNode("environ-pivot")
        pivotNode.setPos(pos[0], pos[1], pos[2])  # Set location of pivot point

        cone = self.loader.loadModel("res/PandaRes/Cone.egg")
        cone.reparentTo(self.render)
        # cone.setPos(pos[0],pos[1],pos[2])
        cone.setColor(col[0], col[1], col[2], 1)
        cone.wrtReparentTo(pivotNode)  # Preserve absolute position
        cone.setPos(0, 0, 0)
        try:
            pivotNode.setHpr(rot[1][1] * rot[0], rot[1][2] * rot[0], rot[1][0] * rot[0])  # Rotates environ around pivot
        except:
            tt = 0
        return cone

    def scatterNeurons(self,maxX,minX,maxY,minY,maxZ,minZ):
        for key,i in self.nvisInterface.neurons.items():
            i.pos = np.array([random.uniform(maxX, minX),random.uniform(maxY, minY),random.uniform(maxZ, minZ)])

    def updateDebugText(self, task):
        self.debugText = ""
        self.debugText = "Camera pos: {0}\n".format(list(self.dummyCameraNode.getPos()))

        for k in self.debugDict.keys():
            self.debugText += k + ": "
            self.debugText += self.debugDict[k] + "\n"
        self.debugTxtBox.text = self.debugText
        return Task.cont

    def showHideLines(self):
        if self.drawLines:
            lineNodes = self.render.findAllMatches("lines")
            for l in lineNodes:
                l.show()
            coneNodes = self.render.findAllMatches("Cone.egg")
            for l in coneNodes:
                l.show()
        else:
            lineNodes = self.render.findAllMatches("lines")
            for l in lineNodes:
                l.hide()

            coneNodes = self.render.findAllMatches("Cone.egg")
            for l in coneNodes:
                l.hide()

    def eventSniffer(self, task):
        if self.nvisInterface.setLayerInfo:
            self.gc.setLayerInfo(self.nvisInterface.layers)
            self.nvisInterface.setLayerInfo = False
        if self.nvisInterface.amIsetingDisplayText:
            self.gc.setTextInput(self.nvisInterface.textToSet)
            self.nvisInterface.amIsetingDisplayText = False
        return task.cont

    def fpsCounter(self, task):
        fr = globalClock.getAverageFrameRate()
        print(str(fr))
        return task.cont


    def start(self):
        ShowBase.__init__(self)

        self.fps = 0
        self.PrevTime = 0

        self.picker = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()

        # mouse dragging
        self.disableMouse()

        self.maxPullingDistance = 5
        self.maxPushingingDistance = 50

        # walls between neurons, this is in panda 3D space units
        self.hardPushingWallDistance = 10

        self.softPushingWallDistance = 30
        self.softPushingWallStrenght = 0.0001

        self.setBackgroundColor(0.2, 0.2, 0.2)

        self.amIPushing = False
        self.showPushingDistance = False

        # debug text, if we want to display text in debug just add to the dictionary
        self.debugText = ""
        self.debugDict = {"H": "debug text"}

        self.debugTxtBox = OnscreenText(text=self.debugText, frame=(0.5, 0.5, 0.5, 1), align=TextNode.ALeft, pos=(-1.6, -0.8), scale=0.03)
        self.debugTxtBox.setColor(0.8, 0.8, 0.8, 1)

        self.taskMgr.add(self.updateDebugText, 'Update debug text')

        #window prop.
        properties = WindowProperties()
        properties.setSize(1300,800)
        self.win.requestProperties(properties)

        # some fancy light
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor(Vec4(0.7, 0.7, 0.7, 1))
        self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLightNodePath)

        mainLight = DirectionalLight("main light")
        self.mainLightNodePath = self.render.attachNewNode(mainLight)
        # Turn it around by 45 degrees, and tilt it down by 45 degrees
        self.mainLightNodePath.setHpr(45, -45, 0)
        self.render.setLight(self.mainLightNodePath)
        self.render.setShaderAuto()


        # make the floor
        floor = self.loader.loadModel("res/PandaRes/Square.egg")
        floor.reparentTo(self.render)
        floor.setPos(0, 0, -10)
        floor.setColor(0.5, 0.5, 0.7, 1)
        floor.setP(-90)
        floor.setScale(50)

        #make the spheres that shows the hard and soft wall
        self.sphereDim = 3.2808398950131235*2

        self.softWallSphere = self.loader.loadModel("res/PandaRes/Sphere_HighPoly.egg")
        self.softWallSphere.reparentTo(self.render)
        #the sphere has a dim of 3.2808398950131235 for some reason so we shrink it
        self.softWallSphere.setScale((1/(self.sphereDim*2))*self.softPushingWallDistance*2)
        self.softWallSphere.setPos(5,5,5)
        self.softWallSphere.setTransparency(1)
        self.softWallSphere.setColor(0.5,1,0.5,0.5)

        self.hardWallSphere = self.loader.loadModel("res/PandaRes/Sphere_HighPoly.egg")
        self.hardWallSphere.reparentTo(self.render)
        self.hardWallSphere.setScale((1/(self.sphereDim*2))*self.hardPushingWallDistance*2)
        self.hardWallSphere.setPos(5,5,5)
        self.hardWallSphere.setTransparency(1)
        self.hardWallSphere.setColor(0.2,5,0.2,0.5)

        self.hardWallSphere.hide()
        self.softWallSphere.hide()

        #we set the positions of neurons in 3d space according to the names eg. A1_1_1 A layer and position [1,1] in the layer

        #self.nvisInterface.setNeuronLayersIn3D()

        # set the camera
        # create a dummy node for camera movment
        self.dummyCameraNode = self.render.attachNewNode("dummyNode")

        self.camera.reparentTo(self.dummyCameraNode)
        self.dummyCameraNode.setPos(0, 0, 100)

        # Tilt the camera down
        self.dummyCameraNode.setP(-90)

        #key controler and camera controler
        kc = KeyControler(self)
        kc.checkKeys()

        #activate the camera controler
        kc.mouseRayControler()

        # Gui
        self.gc = guiControler(self)

        self.gc.rightsideTabGui()
        self.gc.selectionGui()
        self.gc.makeTextInputTab()

        # self.selectionGui()
        self.drawNeurons()



        #update neurons every frame
        self.taskMgr.add(self.updateNeurons, 'Update neurons and cons')

        #self.taskMgr.add(self.fpsCounter, 'fps')

        #update selected neuron text every frame
        self.taskMgr.add(self.gc.updateSelectedNeuronText, 'Update selected neuron text')

        self.taskMgr.add(self.drawBorders, 'draw pushing borders')
        self.taskMgr.add(self.pushNeuronsOutOfProximity, 'draw pushing borders')
        self.taskMgr.add(self.eventSniffer, 'event sniffer')

        #self.setFrameRateMeter(True)

    def __init__(self, nvisInterface):

        # interface for neurovis to communicate with neurovis3D
        self.correlationMethod = None
        self.nvisInterface = nvisInterface
        self.clickedNeuron = None
        self.drawLines = True
        self.layersToSimplify = []
