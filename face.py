import cv2
import dlib
import time

from landmarks import LandmarksMap


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib_face_landmarks.dat')


FAST_FACE_WIDTH = 120  # px, dlib can detect min 80 px width


class Face:

    def __init__(self, img):
        self._img = img
        self._det = None
        self.landmarks_map = None
        self.fast_detected = False

    @classmethod
    def from_frame(cls, img, prev_face=None):
        face = cls(img)
        start_t = time.time()

        if prev_face:
            print('Attempting fast detection...')
            face._det = face._detect_face_fast(img, prev_face)

        if not face._det:
            # if we did not detect a face fast
            print('Attempting slow detection...')
            face._det = face._detect_face(img)

        face._detect_landmarks()
        print('Processed face in {} sec'.format(time.time() - start_t))
        return face

    def _detect_face(self, img):
        dets = detector(img, 0)

        if len(dets) > 1:
            raise ValueError(f'Detected {len(dets)} faces instead of 1')

        if not dets:
            return None

        return dets[0]

    def _detect_face_fast(self, img, prev_face):
        # we want to use prev face detection to crop the image
        # and speed up processing
        if not prev_face.det:
            return None

        img = img.copy()

        # pad a bit so we leave some wiggle room
        pad_horiz = int(prev_face.det.width() * 0.2)
        pad_vert = int(prev_face.det.height() * 0.2)
        x1 = max(prev_face.det.left() - pad_horiz, 0)
        y1 = max(prev_face.det.top() - pad_vert, 0)
        x2 = min(prev_face.det.right() + pad_horiz, img.shape[1])
        y2 = min(prev_face.det.bottom() + pad_vert, img.shape[0])
        # crop the face out of image
        img = img[y1:y2, x1:x2]

        # resize to be smaller and faster to process
        scale = FAST_FACE_WIDTH / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # now we can invoke regular face detection on a smaller image
        det = self._detect_face(img)
        if not det:
            return None

        # need to map coordinates back to original frame
        orig_left = x1 + int(det.left() / scale)
        orig_top = y1 + int(det.top() / scale)
        orig_right = orig_left + int(det.width() / scale)
        orig_bottom = orig_top + int(det.height() / scale)

        det = dlib.rectangle(orig_left, orig_top, orig_right, orig_bottom)
        self.fast_detected = True
        return det

    def _detect_landmarks(self):
        if not self._det:
            return None

        shape = predictor(self._img, self._det)
        self.landmarks_map = LandmarksMap.from_dlib(shape)

    def debug_draw(self, img=None):
        if img is None:
            img = self._img

        if self._det:
            # face rectangle
            x1 = self._det.left()
            y1 = self._det.top()
            x2 = self._det.right()
            y2 = self._det.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            self.mapper.debug_draw(img)

        return img

    @property
    def det(self):
        if not self._det:
            raise ValueError('Face not detected')
        return self._det
