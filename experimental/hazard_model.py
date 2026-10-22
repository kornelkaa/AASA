import torch
import torch.nn as nn
from typing import Dict, Any

class HazardEvaluationNet(nn.Module):
    def __init__(self):
        super(HazardEvaluationNet,self).__init__()

        self.network =  nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class HazardService:
    def __init__(self):
        self.model = HazardEvaluationNet()
        self.model.eval()

        self.class_weights = {
            'person': 1.0,
            'bicycle': 0.9,
            'motorcycle': 0.8,
            'truck': 0.7,
            'bus': 0.7,
            'car': 0.5
        }

    def compute_hazard(self, detection: Dict[str, Any]) -> float:
        """
        Computes numerical hazard score for a single object detection dictionary.
        """

        dist = detection.get('distance_m')
        if dist is None or dist <= 0:
            return 0.0

        label = detection.get('label', '').lower()
        weight = self.class_weights.get(label, 0.3)

        # Bounding box area ratio normalized against standard 1600x900 resolution
        x1, y1, x2, y2 = detection['box']
        box_area_ratio = min(((x2 - x1) * (y2 - y1)) / (1600 * 900), 1.0)

        norm_dist = min(dist / 50.0, 1.0)

        input_tensor = torch.tensor(
        [[1.0 - norm_dist, weight, box_area_ratio]],
            dtype=torch.float32
        )

        with torch.no_grad():
            hazard_score = self.model(input_tensor).item()

        return round(hazard_score, 3)

if __name__ == '__main__':
    service = HazardService()
    test_detection = {
        'label': 'person',
        'confidence': 0.88,
        'box': [100, 200, 300, 500],
        'distance_m': 5.4
    }


    score = service.compute_hazard(test_detection)

    print(f"Target Object: {test_detection['label'].upper()}")
    print(f"Distance: {test_detection['distance_m']} m")
    print(f"Computed Hazard Score: {score * 100:.1f}%")

