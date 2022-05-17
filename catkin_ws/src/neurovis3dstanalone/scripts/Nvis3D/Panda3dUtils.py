import math
import random

import numpy as np
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectEntry import DirectEntry
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectLabel import DirectLabel
from direct.gui.DirectOptionMenu import DirectOptionMenu
from numpy import isnan
from panda3d.core import Vec3, TextNode, LPoint3f, CollisionTraverser, CollisionHandlerQueue, CollisionNode, BitMask32, \
    CollisionRay, Shader, Texture
from direct.gui.DirectGui import *


class KeyControler():
    def __init__(self, sbase, cone=None):
        self.movmentSpeed = 5
        self.sbase = sbase
        self.cone = cone
        self.draggingNeuron = False

        self.keyMap = {
            "up": False,
            "down": False,
            "z-up": False,
            "z-down": False,
            "left": False,
            "right": False,
            "cup": False,
            "cdown": False,
            "cleft": False,
            "cright": False,
            "B": False,
            "cB": False,
            "N": False,
            "cN": False,
            "M": False,
            "cM": False
        }

    def checkKeys(self):
        self.sbase.accept("w", self.updateKeyMap, ["up", True])
        self.sbase.accept("w-up", self.updateKeyMap, ["up", False])
        self.sbase.accept("s", self.updateKeyMap, ["down", True])
        self.sbase.accept("s-up", self.updateKeyMap, ["down", False])
        self.sbase.accept("a", self.updateKeyMap, ["left", True])
        self.sbase.accept("a-up", self.updateKeyMap, ["left", False])
        self.sbase.accept("d", self.updateKeyMap, ["right", True])
        self.sbase.accept("d-up", self.updateKeyMap, ["right", False])
        self.sbase.accept("i", self.updateKeyMap, ["cup", True])
        self.sbase.accept("i-up", self.updateKeyMap, ["cup", False])
        self.sbase.accept("q", self.updateKeyMap, ["z-up", True])
        self.sbase.accept("q-up", self.updateKeyMap, ["z-up", False])
        self.sbase.accept("e", self.updateKeyMap, ["z-down", True])
        self.sbase.accept("e-up", self.updateKeyMap, ["z-down", False])

        self.sbase.accept("k", self.updateKeyMap, ["cdown", True])
        self.sbase.accept("k-up", self.updateKeyMap, ["cdown", False])
        self.sbase.accept("j", self.updateKeyMap, ["cleft", True])
        self.sbase.accept("j-up", self.updateKeyMap, ["cleft", False])
        self.sbase.accept("l", self.updateKeyMap, ["cright", True])
        self.sbase.accept("l-up", self.updateKeyMap, ["cright", False])

        self.sbase.accept("b", self.updateKeyMap, ["B", True])
        self.sbase.accept("b-up", self.updateKeyMap, ["B", False])
        self.sbase.accept("n", self.updateKeyMap, ["N", True])
        self.sbase.accept("n-up", self.updateKeyMap, ["N", False])
        self.sbase.accept("m", self.updateKeyMap, ["M", True])
        self.sbase.accept("m-up", self.updateKeyMap, ["M", False])

        self.updateTask = self.sbase.taskMgr.add(self.updateKeys, "update")

    def updateKeyMap(self, controlName, controlState):
        self.keyMap[controlName] = controlState

    def updateKeys(self, task):
        dt = self.sbase.clock.getDt()

        if self.keyMap["up"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(0, 5.0 * dt, 0) * self.movmentSpeed)
        if self.keyMap["down"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(0, -5.0 * dt, 0) * self.movmentSpeed)
        if self.keyMap["left"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(-5.0 * dt, 0, 0) * self.movmentSpeed)
        if self.keyMap["right"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(5.0 * dt, 0, 0) * self.movmentSpeed)
        if self.keyMap["z-up"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(0, 0, 5.0 * dt) * self.movmentSpeed)
        if self.keyMap["z-down"]:
            self.sbase.dummyCameraNode.setPos(
                self.sbase.dummyCameraNode.getPos() + Vec3(0, 0, -5.0 * dt) * self.movmentSpeed)

        if self.keyMap["cup"]:
            self.sbase.dummyCameraNode.setHpr(
                self.sbase.dummyCameraNode.getHpr() + Vec3(0, 5.0 * dt, 0) * self.movmentSpeed)
        if self.keyMap["cdown"]:
            self.sbase.dummyCameraNode.setHpr(
                self.sbase.dummyCameraNode.getHpr() + Vec3(0, -5.0 * dt, 0) * self.movmentSpeed)
        if self.keyMap["cleft"]:
            self.sbase.dummyCameraNode.setHpr(
                self.sbase.dummyCameraNode.getHpr() + Vec3(5.0 * dt, 0, 0) * self.movmentSpeed)
        if self.keyMap["cright"]:
            self.sbase.dummyCameraNode.setHpr(
                self.sbase.dummyCameraNode.getHpr() + Vec3(-5.0 * dt, 0, 0) * self.movmentSpeed)

        if self.keyMap["B"] == True:
            self.cone.setH(self.cone.getH() + dt * 10)
        if self.keyMap["N"] == True:
            self.cone.setP(self.cone.getP() + dt * 10)
        if self.keyMap["M"] == True:
            self.cone.setR(self.cone.getR() + dt * 10)

        return task.cont

    def mouseClick(self):
        print('mouse click')
        # check if we have access to the mouse
        if self.sbase.mouseWatcherNode.hasMouse():

            # get the mouse position
            mpos = self.sbase.mouseWatcherNode.getMouse()

            # set the position of the ray based on the mouse position
            self.sbase.pickerRay.setFromLens(self.sbase.camNode, mpos.getX(), mpos.getY())
            self.sbase.picker.traverse(self.sbase.render)

            self.sbase.debugDict["mouse click pos"] = str(mpos)
            self.sbase.debugDict["mouse click direction"] = str(self.sbase.pickerRay.getDirection())
            self.sbase.debugDict["mouse click origin"] = str(self.sbase.pickerRay.getOrigin())

            # if we have hit something sort the hits so that the closest is first and highlight the node
            if self.pq.getNumEntries() > 0:
                self.pq.sortEntries()
                pickedObj = self.pq.getEntry(0).getIntoNodePath()

                parent = pickedObj.getParent()
                parent.setColor(0.2, 0.3, 0.7, 1)
                neuronName = pickedObj.getParent().getParent().getName()
                neuron = self.sbase.nvisInterface.getNeuronByName(neuronName)
                self.sbase.clickedNeuron = neuron
                print('click on ' + neuronName)
                self.sbase.gc.SelectedNeuronData.show()

                # dragging
                pFrom = LPoint3f()
                pTo = LPoint3f()
                self.sbase.camLens.extrude(mpos, pFrom, pTo)
                entry0 = self.pq.get_entry(0)
                hitPos = entry0.get_surface_point(self.sbase.render)
                rayFromWorld = self.sbase.render.get_relative_point(self.sbase.camera, pFrom)
                self.oldPickingDist = (hitPos - rayFromWorld).length()
                self.deltaDist = (self.sbase.clickedNeuron.model.get_pos(self.sbase.render) - hitPos)
                self.draggingNeuron = True


            else:
                self.sbase.gc.SelectedNeuronData.hide()
                self.sbase.clickedNeuron = None

    def mouseRealease(self):
        print('mouse release')
        self.draggingNeuron = False

    def neuronDragger(self, task):
        if self.draggingNeuron:
            if self.sbase.clickedNeuron != None:
                if not self.sbase.clickedNeuron.model.isHidden():
                    if self.sbase.mouseWatcherNode.hasMouse():
                        # get the mouse position
                        pMouse = self.sbase.mouseWatcherNode.getMouse()
                        #
                        pFrom = LPoint3f()
                        pTo = LPoint3f()
                        if self.sbase.camLens.extrude(pMouse, pFrom, pTo):
                            # Transform to global coordinates
                            rayFromWorld = self.sbase.render.get_relative_point(self.sbase.camera, pFrom)
                            rayToWorld = self.sbase.render.get_relative_point(self.sbase.camera, pTo)
                            # keep it at the same picking distance
                            direction = (rayToWorld - rayFromWorld).normalized()
                            direction *= self.oldPickingDist
                            self.sbase.clickedNeuron.set3DPos(rayFromWorld + direction + self.deltaDist)
                            if self.sbase.clickedNeuron.layer != "":
                                # we update layer bounds here
                                layer = self.sbase.nvisInterface.layers[self.sbase.clickedNeuron.layer]
                                layer.updateBoundsByNeuron()
        return task.cont

    def mouseRayControler(self):
        self.sbase.picker.showCollisions(self.sbase.render)
        self.pq = CollisionHandlerQueue()

        self.sbase.pickerNode = CollisionNode('mouseRay')
        self.sbase.pickerNP = self.sbase.camera.attachNewNode(self.sbase.pickerNode)
        self.sbase.pickerNode.setFromCollideMask(BitMask32.bit(1))

        self.sbase.pickerRay = CollisionRay()
        self.sbase.pickerNode.addSolid(self.sbase.pickerRay)
        self.sbase.picker.addCollider(self.sbase.pickerNP, self.pq)

        self.sbase.accept("mouse1", self.mouseClick)

        self.sbase.accept("mouse1-up", self.mouseRealease)

        self.daggerTask = self.sbase.taskMgr.add(self.neuronDragger, "neuron dragger")


class guiControler():

    def __init__(self, sbase):
        self.inputTextLabel = None
        self.inputTabs = None
        self.movmentSpeed = 5
        self.sbase = sbase
        self.font = self.sbase.loader.loadFont("res/arial.ttf")
        self.rightsideTabs = []
        self.cbStatusShowDist = False

        self.displayTabsDict = {}
        self.displayTabs = None

    def makeTextInputTab(self):
        textInput = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2), frameSize=(-0.5, 0.5, -0.25, 0), pos=(0, 0, -0.8))
        label = DirectLabel(text="text input:",
                            parent=textInput,
                            scale=0.04,
                            pos=(-0, 0, -0.05),
                            text_font=self.font,
                            frameColor=(0, 0, 0, 0),
                            text_align=TextNode.ACenter
                            )
        self.inputTextLabel = DirectLabel(text="",
                                          parent=textInput,
                                          scale=0.04,
                                          pos=(-0, 0, -0.08),
                                          text_font=self.font,
                                          frameColor=(0, 0, 0, 0),
                                          text_align=TextNode.ACenter
                                          )


    def setTextInput(self, text):
        self.inputTextLabel.setText(text)

    def startCorelationMovment(self):
        try:
            step = self.stepEntry.get()
            corCutoff = self.cutoffEntry.get()
            arrSize = self.numElementsEntry.get()
            method = self.corMethodMenu.get()

            step = float(step)
            corCutoff = float(corCutoff)
            arrSize = int(arrSize)

            self.sbase.startCorelationMovment(step, corCutoff, arrSize, method)



        except:
            return

        pass

    def stopCorelationMovment(self):
        self.sbase.stopCorelationMovment()

    def startPushing(self):

        try:

            hwDis = self.hwDistanceEntry.get()
            swDist = self.swDistanceEntry.get()
            swStr = self.swStrEntry.get()

            hwDis = float(hwDis)
            swDist = float(swDist)
            swStr = float(swStr)

            self.sbase.hardPushingWallDistance = hwDis

            self.sbase.softPushingWallDistance = swDist
            self.sbase.softPushingWallStrenght = swStr

            self.sbase.showPushingDistance = self.cbStatusShowDist
            self.sbase.amIPushing = True



        except:
            return

    def stopPushing(self):
        self.sbase.amIPushing = False

    def setStatusForDist(self, status):
        if status:
            self.cbStatusShowDist = True
        else:
            self.cbStatusShowDist = False

    def drawLines(self, status):
        if status:
            self.sbase.drawLines = True
            self.sbase.showHideLines()
        else:
            self.sbase.drawLines = False
            self.sbase.showHideLines()

    def simplifyLayers(self, status, layer):
        if status:
            self.sbase.layersToSimplify.append(layer.name)
        else:
            self.sbase.layersToSimplify.remove(layer.name)
        pass

    def globalCor(self, status):
        if status:
            self.sbase.nvisInterface.globalCorrelation = True
        else:
            self.sbase.nvisInterface.globalCorrelation = False
        pass

    def hideTabs(self, notToHide=""):
        for i in self.rightsideTabs:
            tabName = i.getTag("tabName")
            if tabName != notToHide:
                i.hide()
            else:
                i.show()

    def scatterN(self):
        try:

            maxX = self.xScatterMax.get()
            minX = self.xScatterMin.get()
            maxY = self.yScatterMax.get()
            minY = self.yScatterMin.get()
            maxZ = self.zScatterMax.get()
            minZ = self.zScatterMin.get()

            maxX = float(maxX)
            minX = float(minX)
            maxY = float(maxY)
            minY = float(minY)
            maxZ = float(maxZ)
            minZ = float(minZ)

            self.sbase.scatterNeurons(maxX, minX, maxY, minY, maxZ, minZ)




        except:
            return

    def corelationTab(self, selectorFrame):
        # corelation tab
        tabCorelation = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                    frameSize=(-0.5, 0, -0.8, 0))
        tabCorelation.reparent_to(selectorFrame)
        tabCorelation.setPos(0, 0, -0.11)
        tabCorelation.setTag("tabName", "cor")

        titleLabel = DirectLabel(text="Corelation controls",
                                 parent=tabCorelation,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        stepLabel = DirectLabel(text="Moving step",
                                parent=tabCorelation,
                                scale=0.03,
                                pos=(-0.5, 0, -0.09),
                                text_font=self.font,
                                frameColor=(0, 0, 0, 0),
                                text_align=TextNode.ALeft
                                )
        self.stepEntry = DirectEntry(initialText="0.1",
                                     parent=tabCorelation,
                                     scale=0.03,
                                     frameSize=(0, 3, 0.8, -0.5),
                                     pos=(-0.3, 0, -0.09),
                                     text_font=self.font,
                                     frameColor=(0.1, 0.1, 0.3, 0.3),
                                     text_align=TextNode.ALeft
                                     )
        numElementsLabel = DirectLabel(text="Num. of ele. for correlation:",
                                       parent=tabCorelation,
                                       scale=0.03,
                                       pos=(-0.5, 0, -0.13),
                                       text_font=self.font,
                                       frameColor=(0, 0, 0, 0),
                                       text_align=TextNode.ALeft
                                       )
        self.numElementsEntry = DirectEntry(initialText="10",
                                            parent=tabCorelation,
                                            scale=0.03,
                                            frameSize=(0, 3, 0.8, -0.5),
                                            pos=(-0.13, 0, -0.13),
                                            text_font=self.font,
                                            frameColor=(0.1, 0.1, 0.3, 0.3),
                                            text_align=TextNode.ALeft
                                            )
        cutoffLabel = DirectLabel(text="Corelation cutoff:",
                                  parent=tabCorelation,
                                  scale=0.03,
                                  pos=(-0.5, 0, -0.17),
                                  text_font=self.font,
                                  frameColor=(0, 0, 0, 0),
                                  text_align=TextNode.ALeft
                                  )
        self.cutoffEntry = DirectEntry(initialText="0.8",
                                       parent=tabCorelation,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.25, 0, -0.17),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        corMethodLabel = DirectLabel(text="Correlation metrix:",
                                     parent=tabCorelation,
                                     scale=0.03,
                                     pos=(-0.5, 0, -0.22),
                                     text_font=self.font,
                                     frameColor=(0, 0, 0, 0),
                                     text_align=TextNode.ALeft
                                     )
        self.corMethodMenu = DirectOptionMenu(
            text="SelectedMethod",
            parent=tabCorelation,
            scale=0.035,
            pos=(-0.25, 0, -0.22),
            frameColor=(0.1, 0.1, 0.3, 0.3),
            items=["pearson"],
            initialitem=0,
            highlightColor=(0.65, 0.65, 0.65, 1))

        startCorB = DirectButton(text=("Go"),
                                 scale=.04,
                                 parent=tabCorelation,
                                 pos=(-0.05, 0, -0.79),
                                 frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                 command=self.startCorelationMovment
                                 )
        stopCorB = DirectButton(text=("End"),
                                scale=.04,
                                parent=tabCorelation,
                                pos=(-0.14, 0, -0.79),
                                frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                command=self.stopCorelationMovment
                                )
        corMethodLabel = DirectLabel(text="Global corelation: ",
                                     parent=tabCorelation,
                                     scale=0.03,
                                     pos=(-0.5, 0, -0.3),
                                     text_font=self.font,
                                     frameColor=(0, 0, 0, 0),
                                     text_align=TextNode.ALeft
                                     )
        globalCorCb = DirectCheckButton(
            parent=tabCorelation,
            scale=0.03,
            frameSize=(0, 0.5, 0.8, -0.5),
            pos=(-0.14, 0, -0.3),
            frameColor=(0.1, 0.1, 0.3, 0.3),
            command=self.globalCor,
            indicatorValue=0
        )

        tabCorelation.hide()

        self.rightsideTabs.append(tabCorelation)

    def pushingTab(self, selectorFrame):
        # corelation tab
        tabPushing = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                 frameSize=(-0.5, 0, -0.8, 0))
        tabPushing.reparent_to(selectorFrame)
        tabPushing.setPos(0, 0, -0.11)
        tabPushing.setTag("tabName", "pus")

        titleLabel = DirectLabel(text="Pushing controls",
                                 parent=tabPushing,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        hwDistanceLabel = DirectLabel(text="Hard wall distance:",
                                      parent=tabPushing,
                                      scale=0.03,
                                      pos=(-0.5, 0, -0.09),
                                      text_font=self.font,
                                      frameColor=(0, 0, 0, 0),
                                      text_align=TextNode.ALeft
                                      )
        self.hwDistanceEntry = DirectEntry(initialText="4",
                                           parent=tabPushing,
                                           scale=0.03,
                                           frameSize=(0, 3, 0.8, -0.5),
                                           pos=(-0.1, 0, -0.09),
                                           text_font=self.font,
                                           frameColor=(0.1, 0.1, 0.3, 0.3),
                                           text_align=TextNode.ALeft
                                           )
        swDistanceLabel = DirectLabel(text="soft wall distance:",
                                      parent=tabPushing,
                                      scale=0.03,
                                      pos=(-0.5, 0, -0.13),
                                      text_font=self.font,
                                      frameColor=(0, 0, 0, 0),
                                      text_align=TextNode.ALeft
                                      )
        self.swDistanceEntry = DirectEntry(initialText="30",
                                           parent=tabPushing,
                                           scale=0.03,
                                           frameSize=(0, 3, 0.8, -0.5),
                                           pos=(-0.13, 0, -0.13),
                                           text_font=self.font,
                                           frameColor=(0.1, 0.1, 0.3, 0.3),
                                           text_align=TextNode.ALeft
                                           )
        swStrLabel = DirectLabel(text="soft wall strength:",
                                 parent=tabPushing,
                                 scale=0.03,
                                 pos=(-0.5, 0, -0.17),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        self.swStrEntry = DirectEntry(initialText="0",
                                      parent=tabPushing,
                                      scale=0.03,
                                      frameSize=(0, 3, 0.8, -0.5),
                                      pos=(-0.13, 0, -0.17),
                                      text_font=self.font,
                                      frameColor=(0.1, 0.1, 0.3, 0.3),
                                      text_align=TextNode.ALeft
                                      )

        showDistLabel = DirectLabel(text="show distance on selected n:",
                                    parent=tabPushing,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.21),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )

        self.showDistCb = DirectCheckButton(
            parent=tabPushing,
            scale=0.03,
            frameSize=(0, 3, 0.8, -0.5),
            pos=(-0.13, 0, -0.21),
            frameColor=(0.1, 0.1, 0.3, 0.3),
            command=self.setStatusForDist,
        )

        startDisB = DirectButton(text=("Go"),
                                 scale=.04,
                                 parent=tabPushing,
                                 pos=(-0.05, 0, -0.79),
                                 frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                 command=self.startPushing,
                                 )
        stopDisB = DirectButton(text=("End"),
                                scale=.04,
                                parent=tabPushing,
                                pos=(-0.14, 0, -0.79),
                                frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                command=self.stopPushing,
                                )

        tabPushing.hide()

        self.rightsideTabs.append(tabPushing)

    def miscTab(self, selectorFrame):
        # corelation tab
        tabMisc = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                              frameSize=(-0.5, 0, -0.8, 0))
        tabMisc.reparent_to(selectorFrame)
        tabMisc.setPos(0, 0, -0.11)
        tabMisc.setTag("tabName", "misc")

        titleLabel = DirectLabel(text="Misc",
                                 parent=tabMisc,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        scatterLabel = DirectLabel(text="Scatter neurons (from to cords):",
                                   parent=tabMisc,
                                   scale=0.03,
                                   pos=(-0.5, 0, -0.09),
                                   text_font=self.font,
                                   frameColor=(0, 0, 0, 0),
                                   text_align=TextNode.ALeft
                                   )
        scatterXLabel = DirectLabel(text="From To X:",
                                    parent=tabMisc,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.13),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )
        self.xScatterMax = DirectEntry(initialText="50",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.3, 0, -0.13),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        self.xScatterMin = DirectEntry(initialText="-50",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.2, 0, -0.13),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        scatterYLabel = DirectLabel(text="From To Y:",
                                    parent=tabMisc,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.17),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )
        self.yScatterMax = DirectEntry(initialText="50",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.3, 0, -0.17),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        self.yScatterMin = DirectEntry(initialText="-50",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.2, 0, -0.17),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        scatterZLabel = DirectLabel(text="From To Z:",
                                    parent=tabMisc,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.21),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )
        self.zScatterMax = DirectEntry(initialText="50",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.3, 0, -0.21),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        self.zScatterMin = DirectEntry(initialText="-10",
                                       parent=tabMisc,
                                       scale=0.03,
                                       frameSize=(0, 3, 0.8, -0.5),
                                       pos=(-0.2, 0, -0.21),
                                       text_font=self.font,
                                       frameColor=(0.1, 0.1, 0.3, 0.3),
                                       text_align=TextNode.ALeft
                                       )
        scatterBut = DirectButton(text=("Scatter"),
                                  scale=.04,
                                  parent=tabMisc,
                                  pos=(-0.5, 0, -0.25),
                                  frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                  command=self.scatterN,
                                  text_align=TextNode.ALeft
                                  )
        showDistLabel = DirectLabel(text="hide connections:",
                                    parent=tabMisc,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.29),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )

        self.showDistCb = DirectCheckButton(
            parent=tabMisc,
            scale=0.03,
            frameSize=(0, 0.5, 0.8, -0.5),
            pos=(-0.13, 0, -0.29),
            frameColor=(0.1, 0.1, 0.3, 0.3),
            command=self.drawLines,
            indicatorValue=1
        )

        tabMisc.hide()

        self.rightsideTabs.append(tabMisc)

    def layerTab(self, selectorFrame):
        # corelation tab
        self.tablay = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                  frameSize=(-0.5, 0, -0.8, 0))
        self.tablay.reparent_to(selectorFrame)
        self.tablay.setPos(0, 0, -0.11)
        self.tablay.setTag("tabName", "lay")

        titleLabel = DirectLabel(text="Layers",
                                 parent=self.tablay,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        self.tablay.hide()
        self.rightsideTabs.append(self.tablay)

    def updateActivity(self, task):
        if self.inputTabs != None:
            # iterate through input neurons and shaders
            for t, n in zip(self.afs, self.sbase.nvisInterface.inputNeurons.values()):
                neuronActCols = []
                for i in n.connectionsIn:
                    act = i.fromNeuron.activity
                    col = self.sbase.getColorFromValues(act)
                    neuronActCols.extend(col)
                t.setShaderInput('myArray', neuronActCols)

        return task.cont

    def makeNeuronInputTabs(self, tabInputN):

        neur = self.sbase.nvisInterface.inputNeurons
        if self.inputTabs != None:
            self.inputTabs.destroy()

        self.afs = []

        self.inputTabs = DirectFrame(frameColor=(1, 0, 0, 0),
                                     parent=tabInputN,
                                     frameSize=tabInputN["frameSize"],
                                     pos=(-0, 0, -0),
                                     )

        ySizeOfTab = -0.55
        for nam, c in zip(neur, range(len(neur))):

            inputTab = DirectFrame(frameColor=(1, 0, 0, 0),
                                   parent=self.inputTabs,
                                   frameSize=(-0.5, 0, ySizeOfTab, 0),
                                   pos=(-0, 0, -0.1 + ySizeOfTab * c),
                                   )

            stepLabel = DirectLabel(text="Input Neuron " + nam,
                                    parent=inputTab,
                                    scale=0.03,
                                    pos=(-0.5, 0, -0.09),
                                    text_font=self.font,
                                    frameColor=(0, 0, 0, 0),
                                    text_align=TextNode.ALeft
                                    )

            mybar = DirectScrollBar(
                parent=inputTab,
                scale=0.5,
                pos=(-0.25, 0, -0.12),
            )
            mybar.setSz(0.2)

            frameSizeX = 0.5
            frameSizeY = -0.4

            activityField = DirectFrame(frameColor=(1, 0.4, 0.6, 0),
                                        frameSize=(0, frameSizeX, 0, frameSizeY),
                                        pos=(-0.5, 0, -0.15),
                                        parent=inputTab,
                                        )
            tmpNCountx = len(neur[nam].connectionsIn)
            tmpNCounty = 1
            neuronActCols = []

            tmpGeom = self.sbase.loader.loadModel("./res/PandaRes/Square.egg")

            actFrame = DirectFrame(frameColor=(0.5, random.uniform(0, 1), random.uniform(0, 1), 1),
                                   frameSize=(0, frameSizeX, 0, frameSizeY),
                                   pos=(0.25, 0, -0.23),
                                   parent=activityField,
                                   geom=tmpGeom)
            actFrame.setScale(0.14)

            for i in neur[nam].connectionsIn:
                act = i.fromNeuron.activity
                col = self.sbase.getColorFromValues(act)
                neuronActCols.extend(col)

            if (len(neur[nam].connectionsIn) == 0):
                neuronActCols = [0.5, 0.5, 0.8]
                tmpNCountx = 1

            my_shader = Shader.load(Shader.SL_GLSL, vertex="./res/shaders/vxSh.glsl",
                                    fragment="./res/shaders/frSh.glsl")
            actFrame.setShaderInput('arrSizeX', tmpNCountx)
            actFrame.setShaderInput('arrSizeY', tmpNCounty)
            actFrame.setShaderInput('myArray', neuronActCols)
            actFrame.setShader(my_shader)

            self.afs.append(actFrame)

    def inputTab(self, selectorFrame):
        # corelation tab
        self.tabinput = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                    frameSize=(-0.5, 0, -0.8, 0))
        self.tabinput.reparent_to(selectorFrame)
        self.tabinput.setPos(0, 0, -0.11)
        self.tabinput.setTag("tabName", "inp")

        titleLabel = DirectLabel(text="Input neurons",
                                 parent=self.tabinput,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )

        minimiseBut = DirectButton(text=("refresh"),
                                   scale=.05,
                                   frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                   parent=self.tabinput,
                                   pos=(-0.1, 0, -0.05),
                                   command=self.makeNeuronInputTabs,
                                   extraArgs=[self.tabinput]
                                   )

        self.makeNeuronInputTabs(self.tabinput)
        self.sbase.taskMgr.add(self.updateActivity, "updateActivity")
        self.rightsideTabs.append(self.tabinput)
        self.tabinput.hide()

    def displayTab(self, selectorFrame):

        # corelation tab
        self.tabDisplay = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                      frameSize=(-0.5, 0, -0.8, 0))
        self.tabDisplay.reparent_to(selectorFrame)
        self.tabDisplay.setPos(0, 0, -0.11)
        self.tabDisplay.setTag("tabName", "dis")

        titleLabel = DirectLabel(text="Data dysplays:",
                                 parent=self.tabDisplay,
                                 scale=0.05,
                                 pos=(-0.5, 0, -0.05),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
        self.displayTabs = DirectFrame(frameColor=(1, 0, 0, 0),
                                       parent=self.tabDisplay,
                                       frameSize=self.tabDisplay["frameSize"],
                                       pos=(-0, 0, -0),
                                       )

        # self.addToDisplayTab(1,np.array([[[150,100,80],[100,200,0],[160,220,0]]]).astype(np.uint8))
        # self.addToDisplayTab(2,np.array([[[1,20,70],[0,0,0],[10,70,90]]]).astype(np.uint8))
        # self.addToDisplayTab(1,[3,1])
        # self.addToDisplayTab(2,[3,1])
        # self.updateDisplayTab(1,np.array([[[150,100,80],[100,200,0],[160,220,0]]]).astype(np.uint8))
        # self.updateDisplayTab(2,np.array([[[1,20,70],[0,0,0],[10,70,90]]]).astype(np.uint8))
        self.rightsideTabs.append(self.tabDisplay)

        self.tabDisplay.hide()

    def addToDisplayTab(self, id, arrSize):
        # make a new display tab
        ySizeOfTab = -0.55

        displayTab = DirectFrame(frameColor=(1, 0, 0, 0),
                                 parent=self.displayTabs,
                                 frameSize=(-0.5, 0, ySizeOfTab, 0),
                                 pos=(-0, 0, -0.1 + ySizeOfTab * len(self.displayTabsDict)),
                                 )

        stepLabel = DirectLabel(text="Input Id " + str(id),
                                parent=displayTab,
                                scale=0.03,
                                pos=(-0.5, 0, -0.09),
                                text_font=self.font,
                                frameColor=(0, 0, 0, 0),
                                text_align=TextNode.ALeft
                                )

        frameSizeX = 0.5
        frameSizeY = -0.4

        activityField = DirectFrame(frameColor=(1, 0.4, 0.6, 0),
                                    frameSize=(0, frameSizeX, 0, frameSizeY),
                                    pos=(-0.5, 0, -0.15),
                                    parent=displayTab,
                                    )

        tmpGeom = self.sbase.loader.loadModel("./res/PandaRes/Square.egg")

        # texture magic
        myTexture = Texture("texture name")
        # texture size
        myTexture.setup2dTexture(arrSize[0], arrSize[1], Texture.T_unsigned_byte, Texture.F_rgb)
        myTexture.setMagfilter(Texture.FT_nearest)

        tmpGeom.setTexture(myTexture)

        actFrame = DirectFrame(frameColor=(0.5, random.uniform(0, 1), random.uniform(0, 1), 1),
                               frameSize=(0, frameSizeX, 0, frameSizeY),
                               pos=(0.25, 0, -0.23),
                               parent=activityField,
                               geom=tmpGeom)
        actFrame.setScale(0.14)

        self.displayTabsDict[id] = tmpGeom

    def updateDisplayTab(self, id, arr):
        # update display tab texture
        if id not in self.displayTabsDict:
            return
        buf = arr.astype(np.uint8).T
        buf = buf.tostring()
        tex = self.displayTabsDict[id].getTexture()
        tex.setRamImage(buf)
        self.displayTabsDict[id].setTexture(tex)

    def rightsideTabGui(self):
        offset = -0.5

        self.selectorFrame = DirectFrame(frameColor=(0.3, 0.4, 0.6, 0.2),
                                         frameSize=(-1.5, 0, -0.1, 0),
                                         pos=(1.6, 0, 0.98))
        # minimise tab
        minimiseBut = DirectButton(text=("-"),
                                   scale=.1,
                                   frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
                                   parent=self.selectorFrame,
                                   pos=(-0.05, 0, -0.05),
                                   command=self.hideTabs
                                   )

        # correlation tab
        corBut = DirectButton(
            text=("Cor"),
            scale=.05,
            frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.95 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["cor"],
        )
        inputBut = DirectButton(
            text=("Input"),
            scale=.05,
            frameColor=((0.15 - 0.5, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.82 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["inp"]
        )

        neuronPushingBut = DirectButton(
            text=("Pushing"),
            scale=.05,
            frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.65 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["pus"]
        )
        miscBut = DirectButton(
            text=("Misc"),
            scale=.05,
            frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.48 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["misc"]
        )

        miscBut = DirectButton(
            text=("Displays"),
            scale=.05,
            frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.30 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["dis"]
        )
        miscBut = DirectButton(
            text=("Layers"),
            scale=.05,
            frameColor=((0.15, 0.2, 0.3, 0.6), (0.2, 0.4, 0.4, 0.6), (0.15, 0.2, 0.4, 0.8)),
            parent=self.selectorFrame,
            pos=(-0.1 + offset, 0, -0.05),
            command=self.hideTabs,
            extraArgs=["lay"]
        )

        self.corelationTab(self.selectorFrame)
        self.inputTab(self.selectorFrame)
        self.pushingTab(self.selectorFrame)
        self.miscTab(self.selectorFrame)
        self.displayTab(self.selectorFrame)
        self.layerTab(self.selectorFrame)

    def selectionGui(self):
        self.SelectedNeuronData = DirectFrame(frameSize=(0, 0.5, -0.25, 1), relief=DGG.FLAT,
                                              frameColor=(0.3, 0.4, 0.6, 0.2),
                                              pos=(-1.6, 0, -0.02)
                                              )

        self.selectedNeuronNameLabel = DirectLabel(text="Neuron: ",
                                                   parent=self.SelectedNeuronData,
                                                   scale=0.03,
                                                   pos=(0, 0, 0.9),
                                                   text_font=self.font,
                                                   frameColor=(0, 0, 0, 0),
                                                   text_align=TextNode.ALeft
                                                   )
        self.selectedNeuronActivityLabel = DirectLabel(text="Activity: ",
                                                       parent=self.SelectedNeuronData,
                                                       scale=0.03,
                                                       pos=(0, 0, 0.86),
                                                       text_font=self.font,
                                                       frameColor=(0, 0, 0, 0),
                                                       text_align=TextNode.ALeft
                                                       )
        self.selectedNeuronConnectionsOutLabel = DirectLabel(text="ConnectionsOut: ",
                                                             parent=self.SelectedNeuronData,
                                                             scale=0.03,
                                                             pos=(0, 0, 0.82),
                                                             text_font=self.font,
                                                             frameColor=(0, 0, 0, 0),
                                                             text_align=TextNode.ALeft,
                                                             text_wordwrap=9)

        self.selectedNeuronConnectionsInLabel = DirectLabel(text="ConnectionsIn: ",
                                                            parent=self.SelectedNeuronData,
                                                            scale=0.03,
                                                            pos=(0, 0, 0.76),
                                                            text_font=self.font,
                                                            frameColor=(0, 0, 0, 0),
                                                            text_align=TextNode.ALeft,
                                                            text_wordwrap=9)

        tmpGeom = self.sbase.loader.loadModel("./res/PandaRes/Square.egg")

        self.selectedConectionIn2D = DirectFrame(frameColor=(0.5, random.uniform(0, 1), random.uniform(0, 1), 1),
                                                 frameSize=(0, 1, 0, 1),
                                                 pos=(0.2475, 0, 0.5),
                                                 parent=self.SelectedNeuronData,
                                                 geom=tmpGeom)
        self.selectedConectionIn2D.setScale(0.14)

        neuronActCols = [0.5, 0.5, 0.8]
        tmpNCountx = 1
        tmpNCounty = 1

        my_shader = Shader.load(Shader.SL_GLSL, vertex="./res/shaders/vxSh.glsl", fragment="./res/shaders/frSh.glsl")
        self.selectedConectionIn2D.setShaderInput('arrSizeX', tmpNCountx)
        self.selectedConectionIn2D.setShaderInput('arrSizeY', tmpNCounty)
        self.selectedConectionIn2D.setShaderInput('myArray', neuronActCols)
        self.selectedConectionIn2D.setShader(my_shader)

        self.selectedConectionOut2D = DirectFrame(frameColor=(0.5, random.uniform(0, 1), random.uniform(0, 1), 1),
                                                  frameSize=(0, 1, 0, 1),
                                                  pos=(0.2475, 0, 0),
                                                  parent=self.SelectedNeuronData,
                                                  geom=tmpGeom)
        self.selectedConectionOut2D.setScale(0.14)

        neuronActCols = [0.5, 0.5, 0.8]
        tmpNCountx = 1
        tmpNCounty = 1

        my_shader = Shader.load(Shader.SL_GLSL, vertex="./res/shaders/vxSh.glsl", fragment="./res/shaders/frSh.glsl")
        self.selectedConectionOut2D.setShaderInput('arrSizeX', tmpNCountx)
        self.selectedConectionOut2D.setShaderInput('arrSizeY', tmpNCounty)
        self.selectedConectionOut2D.setShaderInput('myArray', neuronActCols)
        self.selectedConectionOut2D.setShader(my_shader)

        self.SelectedNeuronData.hide()

    def updateSelectedNeuronText(self, task):
        if self.sbase.clickedNeuron == None:
            return task.cont
        self.selectedNeuronNameLabel.setText("Neuron: " + self.sbase.clickedNeuron.name)
        self.selectedNeuronActivityLabel.setText("Activity: " + str(round(self.sbase.clickedNeuron.activity, 4)))
        connectionsStr = ""
        for i in self.sbase.clickedNeuron.connectionsOut:
            connectionsStr += "[" + i.toNeuron.name + "] "
        self.selectedNeuronConnectionsOutLabel.setText("ConnectionsOut: " + connectionsStr)

        connectionsStr = ""
        for i in self.sbase.clickedNeuron.connectionsIn:
            connectionsStr += "[" + i.fromNeuron.name + "] "
        self.selectedNeuronConnectionsInLabel.setText("ConnectionsIn: " + connectionsStr)

        neuronActCols = []
        for i in self.sbase.clickedNeuron.connectionsIn:
            act = i.fromNeuron.activity
            col = self.sbase.getColorFromValues(act)
            neuronActCols.extend(col)

        if len(self.sbase.clickedNeuron.connectionsIn) > 0:
            self.selectedConectionIn2D.setShaderInput('arrSizeY', 1)
            self.selectedConectionIn2D.setShaderInput('arrSizeX', len(self.sbase.clickedNeuron.connectionsIn))
            self.selectedConectionIn2D.setShaderInput('myArray', neuronActCols)
        else:
            self.selectedConectionIn2D.setShaderInput('arrSizeY', 1)
            self.selectedConectionIn2D.setShaderInput('arrSizeX', 1)
            self.selectedConectionIn2D.setShaderInput('myArray', [0.5, 0.5, 0.8])

        neuronActCols = []
        for i in self.sbase.clickedNeuron.connectionsOut:
            act = i.toNeuron.activity
            col = self.sbase.getColorFromValues(act)
            neuronActCols.extend(col)

        if len(self.sbase.clickedNeuron.connectionsOut) > 0:
            self.selectedConectionOut2D.setShaderInput('arrSizeY', 1)
            self.selectedConectionOut2D.setShaderInput('arrSizeX', len(self.sbase.clickedNeuron.connectionsOut))
            self.selectedConectionOut2D.setShaderInput('myArray', neuronActCols)
        else:
            self.selectedConectionOut2D.setShaderInput('arrSizeY', 1)
            self.selectedConectionOut2D.setShaderInput('arrSizeX', 1)
            self.selectedConectionOut2D.setShaderInput('myArray', [0.5, 0.5, 0.8])

        return task.cont

    def setLayerInfo(self, layers):
        self.layerInfos = DirectFrame(frameColor=(0, 0, 0, 0), frameSize=(-0.5, 0, -0.8, 0), parent=self.tablay)
        posOffset = 0
        for name, val in layers.items():
            tt = 0
            label1 = DirectLabel(text="Layer {0}:".format(name),
                                 parent=self.layerInfos,
                                 scale=0.035,
                                 pos=(-0.5, 0, -0.15 + posOffset),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
            label2 = DirectLabel(text="Num of neurons: {0}".format(len(val.neurons)),
                                 parent=self.layerInfos,
                                 scale=0.035,
                                 pos=(-0.5, 0, -0.20 + posOffset),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )

            label3 = DirectLabel(text="Simplify layer".format(len(val.neurons)),
                                 parent=self.layerInfos,
                                 scale=0.035,
                                 pos=(-0.5, 0, -0.25 + posOffset),
                                 text_font=self.font,
                                 frameColor=(0, 0, 0, 0),
                                 text_align=TextNode.ALeft
                                 )
            simplifyLayerCb = DirectCheckButton(
                parent=self.layerInfos,
                scale=0.03,
                frameSize=(0, 0.5, 0.8, -0.5),
                pos=(-0.3, 0, -0.25 + posOffset),
                frameColor=(0.1, 0.1, 0.3, 0.3),
                command=self.simplifyLayers,
                extraArgs=[val],
                indicatorValue=0
            )
            posOffset -= 0.2


def moveNeuronAndItsControlPointsRelativeToOther(n, pos):
    if (n == None):
        return

    for c in n.connectionsOut:

        # original pos vector
        vec1 = np.array([c.toNeuron.originalpPos[0] - c.fromNeuron.originalpPos[0],
                         c.toNeuron.originalpPos[1] - c.fromNeuron.originalpPos[1],
                         c.toNeuron.originalpPos[2] - c.fromNeuron.originalpPos[2]])
        # pos vector
        vecN = np.array([c.toNeuron.pos[0] - c.fromNeuron.pos[0],
                         c.toNeuron.pos[1] - c.fromNeuron.pos[1],
                         c.toNeuron.pos[2] - c.fromNeuron.pos[2]])

        beforeMoving = n.pos.copy()

        norm = np.linalg.norm(vecN)

        # vec = r2 * vec * self.correlationMoveStep
        vec = np.array([pos[0] - n.pos[0], pos[1] - n.pos[1], pos[2] - n.pos[2]])

        # og control point vector
        vec2 = np.array([c.originalpControlPointPos[0] - c.fromNeuron.originalpPos[0],
                         c.originalpControlPointPos[1] - c.fromNeuron.originalpPos[1],
                         c.originalpControlPointPos[2] - c.fromNeuron.originalpPos[2]])

        # projection of the control point to the vector of neurons
        a = (np.dot(vec1, vec2)) / np.linalg.norm(vec1)
        b = (vec1 / np.linalg.norm(vec1))
        pro = a * b

        ctrlPointTmp = vec2 - pro

        oldDist = np.linalg.norm(np.array(vec1))

        # vec is how we move the neuron

        n.pos[0] += vec[0]
        n.pos[1] += vec[1]
        n.pos[2] += vec[2]

        vec3 = np.array([c.toNeuron.pos[0] - c.fromNeuron.pos[0], c.toNeuron.pos[1] - c.fromNeuron.pos[1],
                         c.toNeuron.pos[2] - c.fromNeuron.pos[2]])

        newDist = np.linalg.norm(np.array(vec3))

        distRelation = newDist / oldDist

        moveCtrlPointTo = distRelation * ctrlPointTmp
        moveCtrlPointTo = np.nan_to_num(moveCtrlPointTo)

        moveControlPointFor = moveCtrlPointTo - ctrlPointTmp
        # move the control point as well

        c.controlPoint[0] = c.originalpControlPointPos[0] + moveControlPointFor[0]
        c.controlPoint[1] = c.originalpControlPointPos[1] + moveControlPointFor[1]
        c.controlPoint[2] = c.originalpControlPointPos[2] + moveControlPointFor[2]

        # dont forget to move thoe control points of other connections as well
        for c2 in n.connectionsIn:
            # original pos vector
            vec1 = np.array([c2.toNeuron.originalpPos[0] - c2.fromNeuron.originalpPos[0],
                             c2.toNeuron.originalpPos[1] - c2.fromNeuron.originalpPos[1],
                             c2.toNeuron.originalpPos[2] - c2.fromNeuron.originalpPos[2]])
            # pos vector
            vecN = np.array([beforeMoving[0] - c2.fromNeuron.pos[0], beforeMoving[1] - c2.fromNeuron.pos[1],
                             beforeMoving[2] - c2.fromNeuron.pos[2]])

            # og control point vector
            vec2 = np.array([c2.originalpControlPointPos[0] - c2.fromNeuron.originalpPos[0],
                             c2.originalpControlPointPos[1] - c2.fromNeuron.originalpPos[1],
                             c2.originalpControlPointPos[2] - c2.fromNeuron.originalpPos[2]])

            # projection of the control point to the vector of neurons
            a = (np.dot(vec1, vec2)) / np.linalg.norm(vec1)
            b = (vec1 / np.linalg.norm(vec1))
            pro = a * b

            ctrlPointTmp = vec2 - pro

            oldDist = np.linalg.norm(np.array(vec1))

            vec3 = np.array(
                [c2.toNeuron.pos[0] - c2.fromNeuron.pos[0], c2.toNeuron.pos[1] - c2.fromNeuron.pos[1],
                 c2.toNeuron.pos[2] - c2.fromNeuron.pos[2]])

            newDist = np.linalg.norm(np.array(vec3))

            distRelation = newDist / oldDist

            moveCtrlPointTo = distRelation * ctrlPointTmp
            moveCtrlPointTo = np.nan_to_num(moveCtrlPointTo)

            moveControlPointFor = moveCtrlPointTo - ctrlPointTmp
            # move the control point as well

            c2.controlPoint[0] = c2.originalpControlPointPos[0] + moveControlPointFor[0]
            c2.controlPoint[1] = c2.originalpControlPointPos[1] + moveControlPointFor[1]
            c2.controlPoint[2] = c2.originalpControlPointPos[2] + moveControlPointFor[2]
