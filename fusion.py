import numpy as np

def calculate_distances(nusc, sample_token, detections):
    sample = nusc.get('sample', sample_token)

    cam_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']

    points,coloring,_ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token = lidar_token,
        camera_token=cam_token
    )

    x_points = points[0, :]
    y_points = points[1, :]
    depths = coloring

    for det in detections:
        x1,y1,x2,y2 = det['box']

        in_box_mask = (
            (x_points >= x1) & (x_points <= x2) &
            (y_points >= y1) & (y_points <= y2) &
            (depths > 0)
        )

        box_depths = depths[in_box_mask]

        if len(box_depths) > 0:
            median_distance = float(np.median(box_depths))
            det['distance_m'] = round(median_distance, 2)

        else:
            det['distance_m'] = None

    return detections

if __name__ == '__main__':
    from dataset import NuScenesDataset
    from detector import SceneDetector
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version='v1.0-mini', dataroot="./data/nuscenes", verbose=False)

    first_scene = nusc.scene[0]
    sample_token = first_scene['first_sample_token']

    dataset = NuScenesDataset(nusc)
    test_image = dataset[0]

    #yolo
    detector = SceneDetector()
    detections = detector.detect(test_image)

    fusion_results = calculate_distances(nusc, sample_token, detections)

    print(f"Sensorfusion evaluation ({len(fusion_results)} object): ")
    for obj in fusion_results:
        dist_str = f"{obj['distance_m']} meter" if obj['distance_m'] else "unknown"

        print(f"-> {obj['label'].upper()}: {dist_str} (Confidence: {obj['confidence']})")