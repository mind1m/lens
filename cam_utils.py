import time
import cv2


def open_cam():
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    if img is None:
        # I have some webcam plugins, so it is 2, in most cases 0
        cam = cv2.VideoCapture(2)

    cam.set(cv2.CAP_PROP_FPS, 30)  # fix fps to 24
    return cam


class FPSTracker:
    def __init__(self):
        self.frames_t = []  # timestamp of last second's frames

    def count_frame(self):
        cur_t = time.time()
        self.frames_t.append(cur_t)
        # remove frames that are more then 1 sec old
        self.frames_t = [t for t in self.frames_t if cur_t - t < 1]

    def draw_fps(self, img):
        img = cv2.putText(
            img, f'FPS: {len(self.frames_t)}', (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 1
        )
        return img

    def count_and_draw(self, img):
        self.count_frame()
        return self.draw_fps(img)
