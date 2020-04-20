import cv2
import dlib
import time
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib_face_landmarks.dat')


def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype='int')
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = np.array([shape.part(i).x, shape.part(i).y])
    # return the list of (x, y)-coordinates
    return coords


FAST_FACE_WIDTH = 120  # px, dlib can detect min 80 px width


class FaceMapper:

    # readable name tp dlib landmark index
    MAPPING = {
        'left_ear': 0,
        'right_ear': 16,
        'left_eye_left': 36,
        'left_eye_right': 39,
        'right_eye_left': 42,
        'right_eye_right': 45,
        'nose_left': 31,
        'nose_right': 35,
        'mouth_left': 48,
        'mouth_right': 64,
        'chin_left': 7,
        'chin_right': 9,
    }

    def __init__(self, dlib_shape):
        self.landmarks = shape_to_np(dlib_shape)

    def get(self, name):
        return self.landmarks[self.MAPPING[name]]

    def debug_draw(self, img):
        for name in self.MAPPING.keys():
            x, y = self.get(name)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(
                img, name, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1
            )
        return img


class Face:

    def __init__(self, img):
        self._img = img
        self._det = None
        self.mapper = None
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

        orig_img = img.copy()  # for debug
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
        self.mapper = FaceMapper(shape)

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
