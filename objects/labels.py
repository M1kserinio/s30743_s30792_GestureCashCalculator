CROP_MARGIN = 0.04  #tyle dokladamy z kazdej strony ramki, zeby nie obciac brzegu banknotu


def read_labels(label_path):
    #linia YOLO: numer_klasy x_srodka y_srodka szerokosc wysokosc (ulamki 0-1)
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append((int(parts[0]), *(float(value) for value in parts[1:])))
    return rows


def to_pixels(x, y, width, height, image_w, image_h):
    return (
        (x - width / 2) * image_w,
        (y - height / 2) * image_h,
        (x + width / 2) * image_w,
        (y + height / 2) * image_h,
    )


def load_labeled_folder(folder, class_names):
    #folder w ukladzie YOLO (images/ + labels/) -> nazwy plikow, zdjecia i prawdziwe ramki w pikselach
    import cv2

    names, images, truths = [], [], []
    for image_path in sorted((folder / "images").glob("*")):
        label_path = folder / "labels" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.exists():
            continue
        image_h, image_w = image.shape[:2]
        names.append(image_path.name)
        images.append(image)
        truths.append([{"box": to_pixels(x, y, w, h, image_w, image_h), "label": class_names[class_id]}
                       for class_id, x, y, w, h in read_labels(label_path)])
    return names, images, truths


def crop_with_margin(image, x1, y1, x2, y2, margin=CROP_MARGIN):
    #tak samo wycinamy przy uczeniu klasyfikatorow, w ocenie potokow i w kamerze
    image_h, image_w = image.shape[:2]
    margin_x = margin * (x2 - x1)
    margin_y = margin * (y2 - y1)
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(image_w, int(x2 + margin_x))
    y2 = min(image_h, int(y2 + margin_y))
    return image[y1:y2, x1:x2]
