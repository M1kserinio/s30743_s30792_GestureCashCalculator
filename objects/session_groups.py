import cv2
import numpy as np

from objects.config import SESSION_SIMILARITY

BACKGROUND_THUMB = 24       #kolaze porownujemy na miniaturach 24x24
SAME_BACKGROUND_DIFF = 10   #srednia roznica pikseli (0-255): to samo tlo daje ~0, rozne tla ponad 20


class SessionGroups:
    def __init__(self, images, collage_boxes):
        self.images = images
        self.collage_boxes = collage_boxes  #ramki dla kolazy mix_banknotow, dla reszty None
        self.same = None  #same[a][b] = True, gdy a i b sa z tej samej sesji

    def groups(self):
        self.same = self.scene_similarity() >= SESSION_SIMILARITY
        collages = [i for i, boxes in enumerate(self.collage_boxes) if boxes is not None]
        if collages:
            self.same[np.ix_(collages, collages)] = self.same_background(collages)

        root = list(range(len(self.images)))

        def find(i):
            while root[i] != i:
                root[i] = root[root[i]]
                i = root[i]
            return i

        #jak A jest z sesji B, a B z sesji C, to cala trojka laduje w jednej grupie
        for a, b in np.argwhere(np.triu(self.same, 1)):
            root[find(a)] = find(b)
        return [find(i) for i in range(len(self.images))]

    def scene_similarity(self):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        transform = weights.transforms()

        normal, mirrored = [], []
        with torch.inference_mode():
            for start in range(0, len(self.images), 32):
                batch = [cv2.imread(str(path)) for path in self.images[start:start + 32]]
                normal.append(model(self.to_batch(batch, transform)))
                mirrored.append(model(self.to_batch([cv2.flip(image, 1) for image in batch], transform)))
        normal = torch.nn.functional.normalize(torch.cat(normal), dim=1).numpy()
        mirrored = torch.nn.functional.normalize(torch.cat(mirrored), dim=1).numpy()

        #lustrzane odbicie ma byc w tej samej grupie co oryginal, wiec porownujemy tez z odbitym
        same = normal @ normal.T
        flipped = normal @ mirrored.T
        return np.maximum(same, np.maximum(flipped, flipped.T))

    def to_batch(self, images, transform):
        import torch
        tensors = [transform(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
                   for image in images]
        return torch.stack(tensors)

    def same_background(self, collages):
        thumbs, masks = [], []
        for i in collages:
            image = cv2.imread(str(self.images[i]))
            thumb = cv2.resize(image, (BACKGROUND_THUMB, BACKGROUND_THUMB), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb.astype(np.float32))
            masks.append(self.background_mask(self.collage_boxes[i]))

        same = np.zeros((len(collages), len(collages)), bool)
        for a in range(len(collages)):
            for b in range(len(collages)):
                common = masks[a] & masks[b]
                if common.sum() >= 30:
                    same[a, b] = np.abs(thumbs[a][common] - thumbs[b][common]).mean() <= SAME_BACKGROUND_DIFF
        return same

    def background_mask(self, boxes):
        mask = np.ones((BACKGROUND_THUMB, BACKGROUND_THUMB), bool)
        for _, x, y, width, height in boxes:
            #o jeden piksel miniatury szerzej, bo brzeg banknotu rozmywa sie przy zmniejszaniu
            x1 = max(0, int((x - width / 2) * BACKGROUND_THUMB) - 1)
            y1 = max(0, int((y - height / 2) * BACKGROUND_THUMB) - 1)
            x2 = int(np.ceil((x + width / 2) * BACKGROUND_THUMB)) + 1
            y2 = int(np.ceil((y + height / 2) * BACKGROUND_THUMB)) + 1
            mask[y1:y2, x1:x2] = False
        return mask
