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

        start_t = time.time()
        self._det = self._detect_face(img)
        self._detect_landmarks()
        print('Processed face in {} sec'.format(time.time() - start_t))

    def _detect_face(self, img):
        dets = detector(img, 0)

        if len(dets) > 1:
            raise ValueError(f'Detected {len(dets)} faces instead of 1')

        if not dets:
            return None

        return dets[0]

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
