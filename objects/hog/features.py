import cv2


def hog_features(bgr_image, image_size=128):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        (image_size, image_size),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )
    return hog.compute(gray).reshape(-1)
