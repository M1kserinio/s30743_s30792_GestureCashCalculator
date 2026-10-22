import cv2


class UI:
    def __init__(self, window_name="Money & Gesture App"):
        self.window_name = window_name
        self.show_help = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def draw_panel(self, img, x1, y1, x2, y2, color=(15, 15, 15), alpha=0.45):
        """Panel półprzezroczysty"""
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 1)

    def draw(self, img, ui_system, counter, operations, gesture_mode, model_name):
        h, w = img.shape[:2]

        if ui_system == 1:
            self._draw_layout_style1(img, w, h, counter, operations, gesture_mode, model_name)
        elif ui_system == 2:
            self._draw_layout_style2(img, w, h, counter, operations, gesture_mode, model_name)
        else:
            self._draw_layout_style3(img, w, h, counter, operations, gesture_mode, model_name)

        if self.show_help:
            self._draw_compact_help(img, w, h, ui_system)

    # Styl 1 to dwie sekcje pol na pol
    def _draw_layout_style1(self, img, w, h, counter, operations, mode, model):
        bar_h = 75
        top_h = h - bar_h
        mid_x = w // 2

        cv2.line(img, (mid_x, 10), (mid_x, top_h - 10), (80, 80, 80), 1)
        cv2.putText(img, "STREFA DLONI", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)
        cv2.putText(img, "STREFA PIENIEDZY", (mid_x + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)

        self.draw_panel(img, 10, top_h + 5, w - 10, h - 10)
        ops_str = ", ".join(operations) if operations else "Brak"
        cv2.putText(img, f"UI 1 | Gest: {mode} | Model: {model} | Wykryto: {ops_str}",
                    (20, top_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        saved = counter.saved_group or "Brak"
        cv2.putText(img, f"ZAPIS: {saved} zl  |  SUMA: {counter.total_value():.2f} zl  |  STATUS: {counter.last_operation_msg}",
                    (20, top_h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1)

    # Styl 2 to kwadraty
    def _draw_layout_style2(self, img, w, h, counter, operations, mode, model):
        self.draw_panel(img, 10, 8, w - 10, 42)
        ops_str = f"[{', '.join(operations)}]" if operations else "[-]"
        saved = counter.saved_group or "Brak"
        text = f"UI 2 | Gest: {mode} {ops_str} | Model: {model} | Zapis: {saved} zl | Suma: {counter.total_value():.2f} zl | {counter.last_operation_msg}"
        cv2.putText(img, text, (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

        box_w, box_h = int(w * 0.24), int(h * 0.32)
        m_x, m_y = 15, 15

        # Lewy róg (LEWA DŁOŃ)
        lx1, ly1 = m_x, h - box_h - m_y
        lx2, ly2 = m_x + box_w, h - m_y
        self.draw_panel(img, lx1, ly1, lx2, ly2)
        cv2.putText(img, "LEWA DLON", (lx1 + 12, ly1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)

        # Prawy róg (PRAWA DŁOŃ)
        rx1, ry1 = w - box_w - m_x, h - box_h - m_y
        rx2, ry2 = w - m_x, h - m_y
        self.draw_panel(img, rx1, ry1, rx2, ry2)
        cv2.putText(img, "PRAWA DLON", (rx1 + 12, ry1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)

    # Styl 3 to pelny ekran
    def _draw_layout_style3(self, img, w, h, counter, operations, mode, model):
        self.draw_panel(img, 10, 8, w - 10, 42)
        ops_str = f"[{', '.join(operations)}]" if operations else "[-]"
        saved = counter.saved_group or "Brak"
        text = f"UI 3 | Gest: {mode} {ops_str} | Model: {model} | ZAPIS: {saved} zl | SUMA: {counter.total_value():.2f} zl | {counter.last_operation_msg}"
        cv2.putText(img, text, (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

    # Komendy
    def _draw_compact_help(self, img, w, h, ui_system):
        pw, ph = 310, 135
        x1, y1 = (w - pw - 10, 50) if ui_system != 1 else (w - pw - 10, h - ph - 85)
        x2, y2 = x1 + pw, y1 + ph

        self.draw_panel(img, x1, y1, x2, y2, color=(10, 10, 10), alpha=0.85)
        cv2.putText(img, "KOMENDY", (x1 + 10, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        col_r = [
            ("R_POINT", "10 zl"),
            ("R_TWO", "20 zl"),
            ("R_SPREAD", "50 zl"),
            ("R_TIGHT", "100 zl"),
            ("R_FIST", "200 zl"),
            ("R_THUMB", "500 zl"),
            ("R_OK", "Wszystko"),
        ]

        col_l = [
            ("L_OK", "Zapis/Reset"),
            ("L_POINT", "Dodaj/Licz"),
            ("L_TWO", "Odejmij/Bez"),
            ("L_FIST", "Menu pomoc"),
            ("L_THUMB", "YOLO/CNN/ResNet"),
            ("L_SPREAD", "Tryb A/B/C"),
            ("L_TIGHT", "Zmien UI"),
        ]

        row_y = y1 + 32
        for i in range(len(col_r)):
            r_g, r_d = col_r[i]
            l_g, l_d = col_l[i]
            cv2.putText(img, f"{r_g:<8} {r_d}", (x1 + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
            cv2.putText(img, f"{l_g:<8} {l_d}", (x1 + 155, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 150), 1)
            row_y += 14

    def show(self, img):
        cv2.imshow(self.window_name, img)

    def close(self):
        cv2.destroyAllWindows()