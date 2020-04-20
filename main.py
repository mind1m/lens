import cv2
import time

from scene import Scene
from lenses import *


def main_cam():
    # I have some webcam plugins, so it is 2, in most cases 0
    cam = cv2.VideoCapture(2)
    scene = Scene(ClownNoseLens)

    frames_t = []  # timestamp of last frames, for fps counter

    while True:
        # get frame
        _, img = cam.read()

        # process - main thing here
        img = scene.process_frame(img)

        # FPS
        cur_t = time.time()
        frames_t.append(cur_t)
        # remove frames that are more then 1 sec old
        frames_t = [t for t in frames_t if cur_t - t < 1]
        cv2.putText(
            img, f'FPS: {len(frames_t)}', (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

        # display
        cv2.imshow('preview', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_cam()
