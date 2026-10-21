from ultralytics import YOLO

class SceneDetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)

    def detect(self, image):
        results = self.model(image, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                label = self.model.names[cls_id]
                confidence = box.conf[0].item() # float(box.conf[0])
                xyxy = [round(coord, 1) for coord in box.xyxy[0].tolist()]

                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': xyxy
                })

        return detections


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from dataset import NuScenesDataset

    nusc = NuScenes(version='v1.0-mini', dataroot="data/nuscenes", verbose=True)

    dataset = NuScenesDataset(nusc)
    test_image = dataset[0]

    detector = SceneDetector()
    detected_objects = detector.detect(test_image)

    print(f"Successful detection, found {len(detected_objects)} objects")
    print("The first five:")

    for obj in detected_objects[:5]:
        print(obj)
