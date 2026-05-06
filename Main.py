import cv2
from GestureModule import Gesture

class Main:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gesture_system = Gesture()

        self.val_left = 12.0
        self.val_right = 15.0

        self.split_left = 0.15
        self.split_right = 0.85

    def draw_ui(self, img, operation):
        h, w, _ = img.shape
        x_left = int(w * self.split_left)
        x_right = int(w * self.split_right)

        cv2.line(img, (x_left, 0), (x_left, h), (255, 255, 255), 2)
        cv2.line(img, (x_right, 0), (x_right, h), (255, 255, 255), 2)

        cv2.putText(img, f"{self.val_left:.2f} zl", (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"{self.val_right:.2f} zl", (int(w * 0.85), 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

        match operation:
            case "L_ADD" | "L_REMOVE" | "L_FULL" | "L_OK":
                cv2.putText(img, f"LEWA: {operation}", (10, h - 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)
            case "R_ADD" | "R_REMOVE" | "R_FULL" | "R_OK":
                cv2.putText(img, f"PRAWA: {operation}", (10, h - 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)
            case "SUM":
                total = self.val_left + self.val_right
                cv2.putText(img, f"SUMA: {total:.2f}", (10, h - 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)
            case "NONE" | _:
                cv2.putText(img, f"NONE", (10, h - 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)


    def run(self):
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img",1200,800)
        while True:
            ok, img = self.cap.read()
            if not ok:
                break

            h, w, _ = img.shape
            x_left_limit = w * self.split_left
            x_right_limit = w * self.split_right

            hand_info = self.gesture_system.detection(img, (x_left_limit, x_right_limit), manual=False)
            operation = self.gesture_system.detect_info(hand_info)

            match operation:
                case "L_ADD":
                    self.val_left += 0.05
                case "L_REMOVE":
                    self.val_left -= 0.05
                case "R_ADD":
                    self.val_right += 0.05
                case "R_REMOVE":
                    self.val_right -= 0.05

            self.draw_ui(img, operation)
            cv2.imshow("img", img)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Main()
    app.run()