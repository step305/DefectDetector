import os

import cv2
from pixellib.instance import custom_segmentation


class DefectDetector:
    def __init__(self, model_file):
        self.model = os.path.abspath(model_file)
        self.segment_image = custom_segmentation()
        self.segment_image.inferConfig(num_classes=1, class_names=["background", "defect"])
        self.segment_image.load_model(self.model)

    def detect(self, img):
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.segment_image.model.detect([new_img])
        result = results[0]
        coords = []
        for class_id, score, roi in zip(result['class_ids'], result['scores'], result['rois']):
            y1, x1, y2, x2 = roi
            coords.append((x1, y1, x2, y2))

        for coord, score in zip(coords, result['scores']):
            x1, y1, x2, y2 = coord
            cv2.rectangle(img,
                          (x1, y1),
                          (x2, y2),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.putText(img,
                        '{:.2f}%'.format(score * 100.0),
                        (x1, y1 - 10),
                        fontFace=cv2.FONT_ITALIC,
                        fontScale=0.4,
                        thickness=1,
                        color=(255, 0, 0))

        report = {'scores': result['scores'],
                  'coords': coords}
        return img, report
