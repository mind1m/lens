import cv2

from scene import Scene
from lenses import GlassesLens, LightningLens, ClownNoseLens
from cam_utils import open_cam, FPSTracker


def main():
    cam = open_cam()
    # send a list of lenses to apply to the Scene
    scene = Scene([LightningLens, GlassesLens])
    fps = FPSTracker()

    while True:
        # get frame
        _, img = cam.read()

        # process - main thing here
        img = scene.process_frame(img, debug=False)

        # FPS
        fps.count_and_draw(img)

        # display
        cv2.imshow('preview', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
