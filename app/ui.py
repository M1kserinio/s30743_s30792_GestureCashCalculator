import cv2


class UI:
    def __init__(self, window_name="img"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)

    def draw(self, img, zones, counter, operation):
        self.draw_zones(img, zones)
        self.draw_values(img, counter)
        self.draw_operation(img, counter, operation)

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

    def draw_operation(self, img, counter, operation):
        match operation:
            case "L_ADD" | "L_REMOVE" | "L_FULL" | "L_OK":
                text = f"LEWA: {operation}"
            case "R_ADD" | "R_REMOVE" | "R_FULL" | "R_OK":
                text = f"PRAWA: {operation}"
            case "SUM":
                text = f"SUMA: {counter.total():.2f}"
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
