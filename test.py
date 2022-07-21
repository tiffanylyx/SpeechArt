from direct.showbase.ShowBase import ShowBase
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from panda3d.core import lookAt
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
from panda3d.core import Texture, GeomNode
from panda3d.core import PointLight, DirectionalLight, AmbientLight,AntialiasAttrib
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3
import sys
import os
from math import pi, sin, cos
from utils import *
import numpy as np
from direct.task import Task
from sympy import *
import random

#base.disableMouse()

from panda3d.core import loadPrcFileData

loadPrcFileData('', 'win-size 1200 400')

base = ShowBase()

# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec

# helper function to make a square given the Lower-Left-Hand and
# Upper-Right-Hand corners
def makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, color_value):

    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('square', format, Geom.UHStatic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    vertex.addData3(x1, y1, z1)
    vertex.addData3(x2, y2, z2)
    vertex.addData3(x3, y3, z3)
    vertex.addData3(x4, y4, z4)

    normal.addData3(normalized(x2-x1, y2-y1, z2-z1))
    normal.addData3(normalized(x3-x1, y3-y1, z3-z1))
    normal.addData3(normalized(x4-x3, y4-y3, z4-z3))
    normal.addData3(normalized(x4-x2, y4-y2, z4-z2))

    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))

    #texcoord.addData2f(0.0, 1.0)
    #texcoord.addData2f(0.0, 0.0)
    #texcoord.addData2f(1.0, 0.0)
    #texcoord.addData2f(1.0, 1.0)

    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(2, 3, 1)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model_sentence = Doc2Vec.load("doc2vec100_text8.model")

count = 0
x_old = count-15 # start time
y_old = 0
z_old = 0
snode = GeomNode('square')
sentence_list = ["get out","this is a nice world","i have compiled a list of basic elements for which parameter values would be needed to created an image","this exploration is hard","the idea is to see how a set of grammar rules can be used to design form in space"]
for sentence in sentence_list:
    x_old = x_old+5
    y_old = 0
    z_old = 0
    [nz_s, nx_s, ny_s] = compute_sent_vec(sentence, model_sentence)
    print("compute_sent_vec",nx_s, ny_s, nz_s)
    word_list = sentence.split(" ")

    res_parts = compute_sent_parts(sentence)
    count_2 = 0
    for word in word_list:
        distance = 0.1 #word speak time
        w = compute_word_length(word)
        l = 2. # how to decide?
        [x1, y1, z1] = solve_start_position(x_old, y_old, z_old, distance, nx_s, ny_s, nz_s) # spread along the sentence vec

        x_old = x1
        y_old = y1
        z_old = z1

        x2 = x1 + w/2 # end time

        color_value = compute_word_vec(word, model, pca2, pca3, pca4, 4) #word vector 4D

        if res_parts[count_2][1] == 'VERB':
            dim = 2
        else: dim = 3

        [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, dim) #word vector 2D or 3D

        y2, z2, x3, y3, z3, x4, y4, z4 = solve_quad(x1,y1,z1,nx,ny,nz,x2,w,l)

        square = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, color_value)
        #square.setColor(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))

        snode.addGeom(square)
        count+=1
        count_2+=1
'''
plight = PointLight('plight')
plight.setColor((0.2, 0.2, 0.2, 1))
plnp = render.attachNewNode(plight)
plnp.setPos(0, 0, 0)
render.setLight(plnp)

plight = PointLight('plight')
plight.setColor((0.2, 0.2, 0.2, 1))
plnp = render.attachNewNode(plight)
plnp.setPos(10, 0, 0)
render.setLight(plnp)

plight = PointLight('plight')
plight.setColor((0.2, 0.2, 0.2, 1))
plnp = render.attachNewNode(plight)
plnp.setPos(0, 10, 0)
render.setLight(plnp)

plight = PointLight('plight')
plight.setColor((0.2, 0.2, 0.2, 1))
plnp = render.attachNewNode(plight)
plnp.setPos(0, 0, 10)
render.setLight(plnp)
'''
alight = AmbientLight('alight')
alight.setColor((0.6, 0.6, 0.6, 0.4))
alnp = render.attachNewNode(alight)
render.setLight(alnp)


#lowPassFilter = AlphaTestAttrib.make(RenderAttrib.MLess,1)
cube = render.attachNewNode(snode)
cube.setDepthWrite(True, 100)
render.setAntialias(AntialiasAttrib.MAuto)
#cube.setAttrib(lowPassFilter)
#cube.setDepthWrite(false)

cube.setTwoSided(True)
angleDegrees = 0
angleDegreesZ = 0
angleRadius = 30
#base.camera.setPos(angleRadius * sin(pi*angleDegrees/ 180), -angleRadius * cos(pi*angleDegrees/ 180),0)
#base.camera.setHpr(angleDegrees, 0,0)

class MyTapper(DirectObject):

    def __init__(self):
        self.testTexture = loader.loadTexture("maps/envir-reeds.png")
        self.accept("1", self.toggleTex)
        self.accept("2", self.toggleLightsSide)
        self.accept("3", self.toggleLightsUp)
        #self.accept("arrow_left", self.moveCameraLeft)
        #self.accept("arrow_right", self.moveCameraRight)

        self.LightsOn = False
        self.LightsOn1 = False
        slight = Spotlight('slight')
        slight.setColor((1, 1, 1, 0))
        lens = PerspectiveLens()
        slight.setLens(lens)
        self.slnp = render.attachNewNode(slight)
        self.slnp1 = render.attachNewNode(slight)

    def moveCameraLeft(self):
        global angleDegrees,angleRadius
        angleDegrees = angleDegrees-5
        base.camera.setPos(angleRadius * sin(pi*angleDegrees/ 180), -angleRadius * cos(pi*angleDegrees/ 180),0)
        base.camera.setHpr(angleDegrees, 0,0)


    def moveCameraRight(self):
        global angleDegrees,angleRadius
        angleDegrees = angleDegrees+5
        base.camera.setPos(angleRadius * sin(pi*angleDegrees/ 180), -angleRadius * cos(pi*angleDegrees/ 180),0)
        base.camera.setHpr(angleDegrees, 0,0)



    def toggleTex(self):
        global cube
        if cube.hasTexture():
            cube.setTextureOff(1)
        else:
            cube.setTexture(self.testTexture)

    def toggleLightsSide(self):
        global cube
        self.LightsOn = not self.LightsOn

        if self.LightsOn:
            render.setLight(self.slnp)
            self.slnp.setPos(cube, 10, -400, 0)
            self.slnp.lookAt(10, 0, 0)
        else:
            render.setLightOff(self.slnp)

    def toggleLightsUp(self):
        global cube
        self.LightsOn1 = not self.LightsOn1

        if self.LightsOn1:
            render.setLight(self.slnp1)
            self.slnp1.setPos(cube, 10, 0, 400)
            self.slnp1.lookAt(10, 0, 0)
        else:
            render.setLightOff(self.slnp1)


t = MyTapper()
base.run()
