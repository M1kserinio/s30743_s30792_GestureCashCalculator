import cv2


class UI:
    def __init__(self, window_name="img"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)

    def draw(self, img, zones, counter, operations, mode):
        self.draw_zones(img, zones)
        self.draw_values(img, counter)
        self.draw_operation(img, operations[0] if operations else "NONE")
        cv2.putText(img, f"TRYB: {mode}  (TAB - zmiana, Q/ESC - wyjscie)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for i, operation in enumerate(operations[1:], start=1):
            cv2.putText(img, operation, (10, img.shape[0] - 50 - 45 * i),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)

    def draw_zones(self, img, zones):
        h = img.shape[0]
        x_left, x_right = int(zones[0]), int(zones[1])
        cv2.line(img, (x_left, 0), (x_left, h), (255, 255, 255), 2)
        cv2.line(img, (x_right, 0), (x_right, h), (255, 255, 255), 2)

    def draw_values(self, img, counter):
        w = img.shape[1]
        cv2.putText(img, f"{counter.left():.2f} zl", (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"{counter.right():.2f} zl", (int(w * 0.85), 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

    def draw_operation(self, img, operation):
        match operation:
            case "L_SPREAD" | "L_FIST" | "L_TIGHT" | "L_POINT" | "L_TWO" | "L_THUMB" | "L_OK":
                text = f"LEWA: {operation}"
            case "R_SPREAD" | "R_FIST" | "R_TIGHT" | "R_POINT" | "R_TWO" | "R_THUMB" | "R_OK":
                text = f"PRAWA: {operation}"
            case "NONE" | _:
                text = "NONE"

        h = img.shape[0]
        cv2.putText(img, text, (10, h - 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 200, 255), 3)

    def show(self, img):
        cv2.imshow(self.window_name, img)

    def quit_pressed(self):
        return cv2.waitKey(1) == ord("q")

    def close(self):
        cv2.destroyAllWindows()
