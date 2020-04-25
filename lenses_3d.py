import cv2
import time
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import *

from estimator_3d import Estimator3D


CHROMAKEY = (0, 0.69, 0.25)
CHROMAKEY_INT_BGR = (64, 176, 0)


class Panda3dApp(ShowBase):

    def _line(self, x, y, z, color):
        lines = LineSegs()
        lines.setColor(color)
        lines.moveTo(0, 0, 0)
        lines.drawTo(x, y, z)
        lines.setThickness(5)
        node = lines.create()
        np = NodePath(node)
        np.reparentTo(self.render)

    def _coords(self):
        self._line(5, 0, 0, (0, 0, 1, 1))
        self._line(0, 5, 0, (0, 1, 0, 1))
        self._line(0, 0, 5, (1, 0, 0, 1))

    def __init__(self, debug=False):
        ShowBase.__init__(self)

        self.obj = self.loader.loadModel('data/cap.3ds')
        self.obj.reparentTo(self.render)
        self.obj.setColorScale((0.1, 0.1, 1, 1))
        self.obj.setPos(-1, 0, 1)
        self.obj.setScale(0.01)
        self.obj.setP(90)
        self.obj.setH(90)

        self.head = self.loader.loadModel('data/head.obj')
        self.head.setColor(*CHROMAKEY)
        self.head.reparentTo(self.render)
        self.head.setPos(-4.1, 0, -7.5)
        self.head.setScale(0.2)
        self.head.setH(90)

        self.cam.reparentTo(self.render)
        self.cam.setPos(10, 0, 0)
        self.cam.lookAt(0, 0, 0)
        self.camLens.setFov(50)

        self.setBackgroundColor(*CHROMAKEY)

        if debug:
            self._coords()

        light = PointLight('plight')
        light.setColor((0.8, 0.8, 0.8, 1))
        light_np = self.render.attachNewNode(light)
        light_np.setPos(5, 0, 2)
        self.obj.setLight(light_np)
        if debug:
            self.head.setLight(light_np)

        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight.upcastToPandaNode())
        self.obj.setLight(alnp)


class Base3DLens:

    def __init__(self, debug=False):
        # if not debug:
        #     loadPrcFileData("", "window-type offscreen" ) # Spawn an offscreen buffer
        self.debug = debug
        self.panda3d_app = Panda3dApp(debug)
        self._estimator_3d = None

    def overlay(self, face_img, landmarks_map):
        self._estimator_3d = Estimator3D(
            landmarks_map, face_img.shape[1], face_img.shape[0]
        )

        # debug purposes
        if self.debug:
            face_img = landmarks_map.debug_draw_3d(self._estimator_3d, face_img)

        self._render()
        rendered_img = self._screenshot()

        return self._combine_3d_2d(face_img, rendered_img)

    def _screenshot(self):
        # get getScreenshot as Texture, could be sped up using texture buffer?
        dr = self.panda3d_app.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()

        # Texture to opencv
        bytes_str = tex.getRamImageAs("BGR").getData()
        img = np.fromstring(bytes_str, np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
        img = cv2.flip(img, 0)  # dunno why but it is inverted in texture

        return img

    def _render(self):
        start_t = time.time()

        self.panda3d_app.cam.setPos(*self._estimator_3d.get_cam_pos())
        self.panda3d_app.cam.lookAt(*self._estimator_3d.get_cam_target_pos())
        self.panda3d_app.cam.setR(self._estimator_3d.get_cam_roll() - 90)

        self.panda3d_app.graphicsEngine.renderFrame()

        print('Rendered frame in {} sec'.format(time.time() - start_t))

    def _crop_chroma(self, img):
        start_t = time.time()

        chroma_pixel = np.array(CHROMAKEY_INT_BGR)
        # same size as img, bool
        # True when non-chromakey pixel
        mask = (img != chroma_pixel).all(axis=2).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))

        # add alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        # Then assign the mask to the last channel of the image
        img[:, :, 3] = mask * 255

        res = img[y:y + h, x:x + w, :]
        print('Crop took {}'.format(time.time()-start_t))

        return res

    def _blend(self, orig_img, lens_img, center_x, center_y):
        # blend lens_img into orig_img centring it at coordinates
        # TODO this could break near borders

        x1 = int(center_x - lens_img.shape[1] / 2)
        y1 = int(center_y - lens_img.shape[0] / 2)

        alpha_lens = lens_img[:, :, 3] / 255.0
        alpha_orig = 1.0 - alpha_lens

        x2 = x1 + lens_img.shape[1]
        y2 = y1 + lens_img.shape[0]

        for c in range(0, 3):
            orig_img[y1:y2, x1:x2, c] = (
                alpha_lens * lens_img[:, :, c] +
                alpha_orig * orig_img[y1:y2, x1:x2, c]
            )

        return orig_img


    def _combine_3d_2d(self, face_img, rendered_img):
        # crop rendered_img to item only
        rendered_img = self._crop_chroma(rendered_img)

        x, y = self._estimator_3d.project_to_2d(-1, 0, 1.5)
        rendered_img = self._blend(face_img, rendered_img, x, y)

        return rendered_img
