import time

from face import Face
from landmarks import smooth_landmarks
from lenses import GlassesLens


class Scene:

    KEEP_LAST = 20  # the bigger the smoother and laggier

    def __init__(self, lense_classes):
        self._lenses = [kls() for kls in lense_classes]
        self._last_slow_face = None
        self._last_faces = []

    def process_frame(self, img, debug=False):
        start_t = time.time()

        # get prev face
        face = Face.from_frame(img, self._last_slow_face)
        if not face.fast_detected:
            self._last_slow_face = face

        if face.det:
            if len(self._last_faces) >= self.KEEP_LAST:
                self._last_faces = self._last_faces[-self.KEEP_LAST:]
            self._last_faces.append(face)

        if debug:
            img = face.debug_draw(img)

        # to avoid wiggle, smooth last faces' landmarks
        smoothed_landmarks_map, rapid_movement = smooth_landmarks(
            [f.landmarks_map for f in self._last_faces]
        )
        if rapid_movement:
            # need to clean the landmarks queue as they are obsolete now
            self._last_faces = [face]

        face_img = face._img.copy()
        for lens in self._lenses:
            face_img = lens.overlay(face_img, smoothed_landmarks_map)

        print('Processed frame in {} sec'.format(time.time() - start_t))
        return face_img
