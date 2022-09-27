from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import *
from direct.filter.CommonFilters import *
from direct.interval.IntervalGlobal import *

class Application(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setupScene()
        self.setupPostFx()

    def setupScene(self):
        self.panda = Actor("panda", {"walk": "panda-walk"})
        self.panda.reparentTo(render)
        self.panda.loop("walk")

        cm = CardMaker("plane")
        cm.setFrame(-10, 10, -10, 10)
        plane = render.attachNewNode(cm.generate())
        plane.setP(270)

        self.cam.setPos(0, -40, 6)
        ambLight = AmbientLight("ambient")
        ambLight.setColor(Vec4(0.2, 0.1, 0.1, 1.0))
        ambNode = render.attachNewNode(ambLight)
        render.setLight(ambNode)
        dirLight = DirectionalLight("directional")
        dirLight.setColor(Vec4(0.1, 0.4, 0.1, 1.0))
        dirNode = render.attachNewNode(dirLight)
        dirNode.setHpr(60, 0, 90)
        render.setLight(dirNode)

        pntLight = PointLight("point")
        pntLight.setColor(Vec4(0.8, 0.8, 0.8, 1.0))
        pntNode = render.attachNewNode(pntLight)
        pntNode.setPos(0, 0, 15)
        self.panda.setLight(pntNode)

        sptLight = Spotlight("spot")
        sptLens = PerspectiveLens()
        sptLight.setLens(sptLens)
        sptLight.setColor(Vec4(1.0, 1.0, 1.0, 1.0))
        sptNode = render.attachNewNode(sptLight)
        sptNode.setPos(-10, -10, 20)
        sptNode.lookAt(self.panda)
        render.setLight(sptNode)

        render.setShaderAuto()

    def setupPostFx(self):
        self.filters = CommonFilters(self.win, self.cam)

        switch = Sequence(Func(self.filters.setBloom, size = "large"), Wait(3), Func(self.filters.delBloom),
                        Func(self.filters.setCartoonInk, 2), Wait(3), Func(self.filters.delCartoonInk),
                        Func(self.filters.setBlurSharpen, 0), Wait(3), Func(self.filters.delBlurSharpen),
                        Func(self.filters.setBlurSharpen, 2), Wait(3), Func(self.filters.delBlurSharpen),
                        Func(self.filters.setInverted), Wait(3), Func(self.filters.delInverted))
        switch.loop()
app = Application()
app.run()
