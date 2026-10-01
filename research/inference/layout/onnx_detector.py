"""PP-DocLayoutV3 inference with the exported graph and original crop rules."""

from types import SimpleNamespace

import cv2
import numpy as np

from research.inference.backends.onnx_runtime import cpu_session

LABELS = ('measure_tab', 'measure_notation', 'measure_both', 'tempo_region',
          'clef_region', 'annotation_region')


class OnnxDetector:
    def __init__(self, path):
        self.session = cpu_session(path)

    def predict(self, pages, *, threshold=.25, **kwargs):
        for page in pages:
            # imdecode supports Unicode paths on Windows, unlike cv2.imread.
            image = (page if isinstance(page, np.ndarray) else
                     cv2.imdecode(np.fromfile(page, dtype=np.uint8), cv2.IMREAD_COLOR))
            if image is None:
                raise ValueError(f'Cannot open page: {page}')
            height, width = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_CUBIC)
            inputs = {'image': image.transpose(2, 0, 1)[None].astype(np.float32) / 255.,
                      'im_shape': np.array([[800, 800]], dtype=np.float32),
                      'scale_factor': np.array([[800 / height, 800 / width]], dtype=np.float32)}
            output = self.session.run(['fetch_name_0'], inputs)[0]
            output[:, 2:6] = np.round(output[:, 2:6])
            output = output[(output[:, 1] > threshold) & (output[:, 0] >= 0)]
            output = output[np.argsort(output[:, 6])]
            boxes = []
            for row in output:
                category = int(row[0])
                x1, y1, x2, y2 = row[2:6].tolist()
                coordinate = [int(max(0, x1)), int(max(0, y1)), int(min(width, x2)), int(min(height, y2))]
                if coordinate[2] > coordinate[0] and coordinate[3] > coordinate[1]:
                    boxes.append({'cls_id': category, 'label': LABELS[category],
                                  'score': float(row[1]), 'coordinate': coordinate})
            yield SimpleNamespace(json={'res': {'boxes': boxes}})
