import cv2
import numpy as np


class Conditions:
    #Sztuczne zmiany warunkow nakladane na zdjecia testowe.
    #Ramki sa w pikselach: (klasa, x1, y1, x2, y2). Zmiany, ktore przesuwaja banknot (obrot, perspektywa)
    #albo go czesciowo zaslaniaja, przeliczaja tez ramki, zeby ocena dalej byla uczciwa

    def __init__(self, backgrounds):
        #zdjecia bez banknotow z testu (datasets/brak_banknotu/test) - do podmiany tla i do zaslaniania
        self.backgrounds = backgrounds
        #nazwa folderu -> (czego dotyczy, funkcja)
        self.all = {
            "oryginal": ("bez zmian", self.original),
            "ciemno": ("oswietlenie", self.dark),
            "przeswietlenie": ("oswietlenie", self.overexposed),
            "zolte_swiatlo": ("oswietlenie", self.warm_light),
            "cien": ("oswietlenie", self.shadow),
            "inne_tlo": ("tlo", self.other_background),
            "wzorzyste_tlo": ("tlo", self.busy_background),
            "obrot_15": ("kat", lambda image, boxes, rng: self.rotate(image, boxes, rng, 15)),
            "obrot_35": ("kat", lambda image, boxes, rng: self.rotate(image, boxes, rng, 35)),
            "perspektywa": ("kat", self.tilt),
            "zasloniete_25": ("zasloniecie", lambda image, boxes, rng: self.cover(image, boxes, rng, 0.25)),
            "zasloniete_50": ("zasloniecie", lambda image, boxes, rng: self.cover(image, boxes, rng, 0.5)),
            "zgiecie": ("stan banknotu", self.fold),
            "zagiety_rog": ("stan banknotu", self.dog_ear),
            "brud": ("stan banknotu", self.dirt),
            "rozmycie_ruchu": ("kamera", self.motion_blur),
        }

    def original(self, image, boxes, rng):
        return image, boxes

    def dark(self, image, boxes, rng):
        #wieczor w pokoju: malo swiatla, a kamerka podbija wzmocnienie, wiec rosnie szum
        return to_uint8(image * 0.35 + rng.normal(0, 7, image.shape)), boxes

    def overexposed(self, image, boxes, rng):
        #lampa prosto w banknot: jasne miejsca sie "wypalaja" i gina detale
        return to_uint8(image * 1.6 + 45), boxes

    def warm_light(self, image, boxes, rng):
        #zarowka zamiast dnia: wszystko zolto-pomaranczowe, a nominaly rozrozniamy m.in. po kolorze (BGR)
        return to_uint8(image * np.array([0.65, 0.95, 1.2])), boxes

    def shadow(self, image, boxes, rng):
        #cien (np. reki albo telefonu) z miekka krawedzia przechodzacy przez kadr
        h, w = image.shape[:2]
        angle = rng.uniform(0, np.pi)
        center_x, center_y = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        ys, xs = np.mgrid[0:h, 0:w]
        distance = (xs - center_x) * np.cos(angle) + (ys - center_y) * np.sin(angle)
        mask = 1 / (1 + np.exp(-distance / (0.03 * max(h, w))))
        return to_uint8(image * (1 - 0.6 * mask)[..., None]), boxes

    def other_background(self, image, boxes, rng):
        #banknoty zostaja na miejscu, a reszta kadru to mozaika z 4 kawalkow innych zdjec bez banknotow
        h, w = image.shape[:2]
        tiles = [cover(self.pick(rng), w // 2, h // 2) for _ in range(4)]
        background = cv2.resize(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), (w, h))
        return paste_boxes(background, image, boxes), boxes

    def busy_background(self, image, boxes, rng):
        #kolorowa kratka (obrus, gazeta) - trudne dla modeli patrzacych glownie na kolor
        h, w = image.shape[:2]
        cells = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        background = cv2.resize(cells, (w, h), interpolation=cv2.INTER_NEAREST)
        return paste_boxes(background, image, boxes), boxes

    def rotate(self, image, boxes, rng, degrees):
        #kamera trzymana krzywo. Banknot w ramce przyblizamy elipsa wpisana w ramke (tak robi albumentations),
        #bo prostokat opisany na obroconych rogach ramki bylby duzo wiekszy od samego banknotu
        h, w = image.shape[:2]
        angle = degrees * rng.choice([-1, 1])
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=fill_color(image))
        return rotated, move_boxes(boxes, lambda points: cv2.transform(points[None], matrix)[0], w, h,
                                   outline=ellipse_points)

    def tilt(self, image, boxes, rng):
        #kamera patrzy z ukosa: jedna krawedz kadru "ucieka" w glab i robi sie krotsza
        h, w = image.shape[:2]
        squeeze = 0.15
        source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        target = source.copy()
        side = rng.integers(4)
        if side == 0:  #gora
            target[0, 0] += squeeze * w
            target[1, 0] -= squeeze * w
        elif side == 1:  #dol
            target[3, 0] += squeeze * w
            target[2, 0] -= squeeze * w
        elif side == 2:  #lewa
            target[0, 1] += squeeze * h
            target[3, 1] -= squeeze * h
        else:  #prawa
            target[1, 1] += squeeze * h
            target[2, 1] -= squeeze * h
        matrix = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=fill_color(image))
        return warped, move_boxes(boxes, lambda points: cv2.perspectiveTransform(points[None], matrix)[0], w, h)

    def cover(self, image, boxes, rng, part):
        #czesc banknotu przykryta (palcem, portfelem, innym przedmiotem) - pasek od jednego boku ramki
        #nowa ramka obejmuje tylko widoczna czesc, bo tak zaznaczylby ja czlowiek
        result = image.copy()
        visible = []
        for class_id, x1, y1, x2, y2 in pixel_boxes(boxes, image):
            cut_w, cut_h = round(part * (x2 - x1)), round(part * (y2 - y1))
            side = rng.integers(4)
            if side == 0:
                hidden, kept = (x1, y1, x1 + cut_w, y2), (x1 + cut_w, y1, x2, y2)
            elif side == 1:
                hidden, kept = (x2 - cut_w, y1, x2, y2), (x1, y1, x2 - cut_w, y2)
            elif side == 2:
                hidden, kept = (x1, y1, x2, y1 + cut_h), (x1, y1 + cut_h, x2, y2)
            else:
                hidden, kept = (x1, y2 - cut_h, x2, y2), (x1, y1, x2, y2 - cut_h)
            hx1, hy1, hx2, hy2 = hidden
            if hx2 > hx1 and hy2 > hy1:
                result[hy1:hy2, hx1:hx2] = cover(self.pick(rng), hx2 - hx1, hy2 - hy1)
            visible.append((class_id, *kept))
        return result, visible

    def fold(self, image, boxes, rng):
        #banknot zgiety w poprzek: jedna polowa odchylona od swiatla, przy zgieciu ciemna linia
        result = image.astype(np.float32)
        for _, x1, y1, x2, y2 in pixel_boxes(boxes, image):
            region = result[y1:y2, x1:x2].copy()
            h, w = region.shape[:2]
            if h < 4 or w < 4:
                continue
            ys, xs = np.mgrid[0:h, 0:w]
            along = xs / w if w >= h else ys / h  #zginamy w poprzek dluzszego boku
            crease = rng.uniform(0.35, 0.65)
            light = np.where(along < crease, 1.0, 0.6 + 0.25 * (along - crease) / (1 - crease))
            light -= 0.35 * np.exp(-((along - crease) / 0.02) ** 2)
            result[y1:y2, x1:x2] = region * light[..., None]
        return to_uint8(result), boxes

    def dog_ear(self, image, boxes, rng):
        #zagiety rog: w rogu widac tlo, a obok lezy odwrocony, ciemniejszy trojkat z tylu banknotu
        result = image.copy()
        image_h, image_w = image.shape[:2]
        for _, x1, y1, x2, y2 in pixel_boxes(boxes, image):
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            leg = rng.uniform(0.25, 0.4)
            corner_x, corner_y = (x1, x2)[rng.integers(2)], (y1, y2)[rng.integers(2)]
            step_x = leg * (x2 - x1) * (1 if corner_x == x1 else -1)
            step_y = leg * (y2 - y1) * (1 if corner_y == y1 else -1)
            corner = np.float32([corner_x, corner_y])
            a = np.float32([corner_x + step_x, corner_y])
            b = np.float32([corner_x, corner_y + step_y])
            folded = reflect(corner, a, b)

            #rog po zagieciu = lustrzane odbicie rogu wzgledem linii a-b, przyciemnione i wyblakle
            matrix = cv2.getAffineTransform(np.float32([corner, a, b]), np.float32([folded, a, b]))
            flap = cv2.warpAffine(image, matrix, (image_w, image_h)).astype(np.float32)
            gray = cv2.cvtColor(flap.astype(np.uint8), cv2.COLOR_BGR2GRAY)[..., None]
            flap = to_uint8((0.5 * flap + 0.5 * gray) * 0.75)

            corner_mask = np.zeros((image_h, image_w), np.uint8)
            cv2.fillConvexPoly(corner_mask, np.int32([corner, a, b]), 255)
            flap_mask = np.zeros((image_h, image_w), np.uint8)
            cv2.fillConvexPoly(flap_mask, np.int32([folded, a, b]), 255)
            result[corner_mask > 0] = fill_color(image)
            result[flap_mask > 0] = flap[flap_mask > 0]
            cv2.line(result, tuple(int(v) for v in a), tuple(int(v) for v in b), (40, 40, 40), 1)
        return result, boxes

    def dirt(self, image, boxes, rng):
        #sfatygowany banknot: pozolkly papier, plamy i rysy
        result = image.astype(np.float32)
        for _, x1, y1, x2, y2 in pixel_boxes(boxes, image):
            region = result[y1:y2, x1:x2].copy()
            h, w = region.shape[:2]
            if h < 10 or w < 10:
                continue
            region = region * 0.85 + np.array([20, 45, 60])
            stains = np.zeros((h, w), np.float32)
            for _ in range(rng.integers(4, 9)):
                center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
                axes = (max(1, int(rng.uniform(0.05, 0.18) * w)), max(1, int(rng.uniform(0.05, 0.18) * h)))
                cv2.ellipse(stains, center, axes, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.4, 0.8)), -1)
            stains = cv2.GaussianBlur(stains, (0, 0), max(1.0, 0.02 * w))[..., None]
            region = region * (1 - stains) + np.array([35, 55, 75]) * stains
            for _ in range(rng.integers(2, 5)):
                start = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
                end = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
                cv2.line(region, start, end, (60, 60, 60), 1)
            result[y1:y2, x1:x2] = region
        return to_uint8(result), boxes

    def motion_blur(self, image, boxes, rng):
        #reka z banknotem sie rusza, a kamerka ma dlugi czas naswietlania
        size = 15
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1
        rotation = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), float(rng.uniform(0, 180)), 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (size, size))
        return cv2.filter2D(image, -1, kernel / kernel.sum()), boxes

    def pick(self, rng):
        return self.backgrounds[rng.integers(len(self.backgrounds))]


def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def fill_color(image):
    #kolor "pustego" miejsca po obrocie albo po zagietym rogu - mediana calego zdjecia
    return tuple(float(v) for v in np.median(image.reshape(-1, 3), axis=0))


def pixel_boxes(boxes, image):
    h, w = image.shape[:2]
    return [(class_id, max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))
            for class_id, x1, y1, x2, y2 in boxes]


def paste_boxes(background, image, boxes):
    result = background.copy()
    for _, x1, y1, x2, y2 in pixel_boxes(boxes, image):
        result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return result


def cover(patch, width, height):
    #skaluje kawalek tak, zeby przykryl caly prostokat, i obcina nadmiar (jak "cover" w CSS)
    scale = max(width / patch.shape[1], height / patch.shape[0])
    resized = cv2.resize(patch, (max(width, round(patch.shape[1] * scale)), max(height, round(patch.shape[0] * scale))))
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


def corner_points(x1, y1, x2, y2):
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def ellipse_points(x1, y1, x2, y2, count=72):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.float32(np.stack([(x1 + x2) / 2 + (x2 - x1) / 2 * np.cos(angles),
                                (y1 + y2) / 2 + (y2 - y1) / 2 * np.sin(angles)], axis=1))


def move_boxes(boxes, transform, width, height, outline=corner_points):
    moved = []
    for class_id, x1, y1, x2, y2 in boxes:
        points = transform(outline(x1, y1, x2, y2))
        (new_x1, new_y1), (new_x2, new_y2) = points.min(axis=0), points.max(axis=0)
        area = (new_x2 - new_x1) * (new_y2 - new_y1)
        new_x1, new_y1 = max(0.0, new_x1), max(0.0, new_y1)
        new_x2, new_y2 = min(float(width), new_x2), min(float(height), new_y2)
        #banknot wypchniety w wiekszosci poza kadr nie jest juz do znalezienia
        if new_x2 > new_x1 and new_y2 > new_y1 and (new_x2 - new_x1) * (new_y2 - new_y1) >= 0.5 * area:
            moved.append((class_id, float(new_x1), float(new_y1), float(new_x2), float(new_y2)))
    return moved


def reflect(point, a, b):
    #odbicie punktu wzgledem prostej przechodzacej przez a i b
    direction = (b - a) / np.linalg.norm(b - a)
    foot = a + np.dot(point - a, direction) * direction
    return 2 * foot - point
