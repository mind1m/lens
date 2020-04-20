from face import Face
from lenses import GlassesLens


class Scene:

    KEEP_LAST = 10

    def __init__(self, lense_cls=GlassesLens):
        self._lense = lense_cls()
        self._last_slow_face = None

    def process_frame(self, img):
        # get prev face
        face = Face.from_frame(img, self._last_slow_face)
        if not face.fast_detected:
            self._last_slow_face = face

        # img = face.debug_draw(img)

        complete_img = self._lense.overlay(face)

        return complete_img
