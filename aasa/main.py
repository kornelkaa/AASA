import os
import json
from nuscenes.nuscenes import NuScenes
from aasa.dataset import NuScenesDataset
from aasa.detector import SceneDetector
from aasa.fusion import calculate_distances
from aasa.agent import AutonomousSceneAgent


def run_pipeline(scene_index=0, frame_index=0):
    print("Pipeline starts")


    os.makedirs("../outputs", exist_ok=True)

    # nuscenes
    print("Nuscenes initalizing")
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)

    selected_scene = nusc.scene[scene_index]
    sample_token = selected_scene['first_sample_token']


    # dataset initalization
    print("dataset")
    dataset = NuScenesDataset(nusc)
    image = dataset[frame_index]


    # yolov8
    print("yolov8 detector")
    detector = SceneDetector()
    detections = detector.detect(image)


    # detected picture saved as visualization
    raw_results = detector.model(image, conf=0.15, verbose=False)
    output_image_path = "../outputs/detected_scene.jpg"
    raw_results[0].save(filename=output_image_path)
    print(f"  -> Saved: {output_image_path}")


    # sensorfusion
    print("sensorfusion")
    fusion_results = calculate_distances(nusc, sample_token, detections)


    # ai agent analysis
    print("AI Agent")
    agent = AutonomousSceneAgent()
    decision = agent.analyze_scene(fusion_results)


    # saving to file
    output_json_path = "../outputs/agent_decision.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=4, ensure_ascii=False)
    print("saved")


    # printing
    print(f"RISK LEVEL:  {decision.get('risk_level', 'N/A')}")
    print(f"RECOMMENDED_ACTION:   {decision.get('recommended_action', 'N/A')}")
    print(f"CRITICAL OBJECT:    {decision.get('critical_object', 'N/A')}")
    print(f"REASONING:         {decision.get('reasoning', 'N/A')}")


if __name__ == '__main__':
    run_pipeline(0,0)

