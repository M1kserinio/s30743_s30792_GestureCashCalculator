class CashCounter:
    def __init__(self, use_detector):
        #kwota w strefie = detected (wykryl YOLO) + correction (korekta gestami)
        self.detected_left = 0.0
        self.detected_right = 0.0
        #z YOLO korekta startuje od 0, inaczej doliczaloby 12 i 15 zl do wykrytych pieniedzy
        self.correction_left = 0.0 if use_detector else 12.0
        self.correction_right = 0.0 if use_detector else 15.0

    def set_detected(self, left, right):
        self.detected_left = left
        self.detected_right = right

    def apply(self, operation):
        match operation:
            case "L_ADD":
                self.correction_left += 0.05
            case "L_REMOVE":
                self.correction_left -= 0.05
            case "R_ADD":
                self.correction_right += 0.05
            case "R_REMOVE":
                self.correction_right -= 0.05

    def left(self):
        return self.detected_left + self.correction_left

    def right(self):
        return self.detected_right + self.correction_right

    def total(self):
        return self.left() + self.right()
