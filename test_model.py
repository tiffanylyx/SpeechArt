from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor

from panda3d.core import LVecBase3
from panda3d.core import PointLight, DirectionalLight

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((0.8, 0.8, 0.8, 1))
        directionalLight.setShadowCaster(True, 512, 512)
        dlens = directionalLight.getLens()
        dlens.setFilmSize(41, 21)
        dlens.setNearFar(50, 75)
        directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setHpr(0, -20, 0)
        self.render.setLight(directionalLightNP)

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((0.8, 0.8, 0.8, 1))
        directionalLight.setShadowCaster(True, 512, 512)
        dlens = directionalLight.getLens()
        dlens.setFilmSize(41, 21)
        dlens.setNearFar(50, 75)
        directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setHpr(20,0, 0)
        self.render.setLight(directionalLightNP)


        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        self.pandaActor = self.loader.loadModel("models/panda")
        self.pandaActor.setDepthOffset(1)
        self.render.setShaderAuto()
        self.pandaActor.setScale(0.3, 0.3, 0.3)
        self.pandaActor.setPos(0,0,0)
        self.pandaActor.setShaderAuto()
        #self.pandaActor.setHpr(0,100,0)
        #self.pandaActor.setPosHprScale(LVecBase3(0,0,1),LVecBase3(0,100,0),LVecBase3(0.05, 0.05, 1))
        self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        #self.pandaActor.loop("walk")

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


app = MyApp()
app.run()
