'''
import direct.directbase.DirectStart
from direct.showbase import DirectObject
from panda3d.core import *
from direct.task import Task
import time

class Test(DirectObject.DirectObject):
    def __init__(self):
        self.accept('spam', self.on_spam, ['eggs', 'sausage'])

    async def on_spam(self, a, b, c, d):
        await Task.pause(1.0)
        print("The space key was pressed one second ago!")

        print(a, b, c, d)



test = Test()

while True:
    time.sleep(1)

    messenger.send('spam', ['foo', 'bar'])

base.run()
'''
import time
import direct.directbase.DirectStart
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase,LVecBase3
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task

from panda3d.core import lookAt
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter,GeomTristrips
from panda3d.core import Texture, GeomNode
from panda3d.core import PointLight, DirectionalLight, AmbientLight,AntialiasAttrib
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3
from panda3d.core import loadPrcFileData

import random
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec
def makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):

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

    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))

    color.addData4(1,0,0,1)
    color.addData4(1,0,0,1)
    color.addData4(1,0,0,1)
    color.addData4(1,0,0,1)

    #texcoord.addData2f(0.0, 1.0)
    #texcoord.addData2f(0.0, 0.0)
    #texcoord.addData2f(1.0, 0.0)
    #texcoord.addData2f(1.0, 1.0)
    tris = GeomTristrips(Geom.UHDynamic)
    tris.addVertices(0,1,2,3)
    '''
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(1, 2, 3)
'''
    square = Geom(vdata)
    square.addPrimitive(tris)
    return square
class Test(DirectObject):
    def __init__(self):
        self.accept("FireZeMissiles", self._fireMissiles)

    def _fireMissiles(self):
        x_old_1 = random.random()
        y_old_1 = random.random()
        z_old_1 = random.random()
        x_old_2 = random.random()
        y_old_2 = random.random()
        z_old_2 = random.random()
        x1 = random.random()
        y1 = random.random()
        z1 = random.random()
        x2 = random.random()
        y2 = random.random()
        z2 = random.random()

        square = makeQuad(x_old_1, y_old_1, z_old_1, x_old_2, y_old_2, z_old_2, x1, y1, z1, x2, y2, z2)
        snode = GeomNode('square')
        snode.addGeom(square)
        cube = render.attachNewNode(snode)
        cube.setDepthWrite(True, 100)
        cube.setTwoSided(True)
        print("Missiles fired! Oh noes!")

    # function to get rid of me
    def destroy(self):
        self.ignoreAll()

foo = Test()  # create our test object
#foo.destroy() # get rid of our test object

#del foo

for i in range(10):
    messenger.send("FireZeMissiles") # No missiles fire
    time.sleep(1)
base.run()
