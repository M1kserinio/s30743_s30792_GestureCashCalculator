import cv2
from GestureModule import Gesture

class Main:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gesture_system = Gesture()

    def draw_ui(self, img, gesture):
        h, w, _ = img.shape
        text = gesture if gesture else "BRAK GESTU"

        cv2.putText(
            img,
            text,
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 180),
            3
        )

    def run(self):
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img",1200,800)
        while True:
            ok, img = self.cap.read()
            if not ok:
                break

            img = cv2.flip(img, 1) # flipujmy to odrazu bo i tak kamere bedziemyt miec z gory

            gesture = self.gesture_system.detection(img)
            self.gesture_system.detect_info(gesture)

            self.draw_ui(img, gesture)
            cv2.imshow("img", img)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Main()
    app.run()