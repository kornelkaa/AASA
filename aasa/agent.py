import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AutonomousSceneAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Missing the GROQ_API_KEY!")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def build_prompt(self, fusion_results):
        detected_summary = []
        for obj in fusion_results:
            label = obj['label'].upper()
            confidence = int(obj['confidence'] * 100)
            dist = f"{obj['distance_m']} meters" if obj['distance_m'] is not None else "Unknown distance"
            
            detected_summary.append(f"- {label} at {dist} (Detection Confidence: {confidence}%)")

        summary_text = "\n".join(detected_summary)

        prompt = f"""
You are the central decision-making agent of an Autonomous Vehicle.
Analyze the following real-time 3D sensor fusion detections from the front camera and Lidar sensor:

[DETECTED OBJECTS AHEAD]
{summary_text}

[INSTRUCTIONS]
1. Assess the immediate safety risk to the ego-vehicle based on object proximity and type.
2. Determine the most critical object.
3. Select the best recommended driving action.

[OUTPUT FORMAT REQUIREMENT]
Respond strictly with a valid JSON object matching this exact schema:
{{
    "scene_summary": "<Brief 1-2 sentence description of the current scenario>",
    "risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>",
    "critical_object": "<Label and distance of the main hazard, or 'None'>",
    "recommended_action": "<MAINTAIN_SPEED | SLOW_DOWN | BRAKE_IMMEDIATELY | YIELD>",
    "reasoning": "<Short explanation for why this action was chosen>"
}}
"""
        return prompt

    def analyze_scene(self, fusion_results):
        prompt = self.build_prompt(fusion_results)

        response = self.client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        return json.loads(response.choices[0].message.content)

if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from aasa.dataset import NuScenesDataset
    from aasa.detector import SceneDetector
    from aasa.fusion import calculate_distances

    print("--- FULL PIPELINE RUNNING ---")
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)
    
    first_scene = nusc.scene[0]
    sample_token = first_scene['first_sample_token']
    
    # Full pipeline 1.
    dataset = NuScenesDataset(nusc)
    test_image = dataset[0]
    
    # 2. scene detecting
    detector = SceneDetector()
    detections = detector.detect(test_image)
    
    # 3. sensorfusion
    fusion_results = calculate_distances(nusc, sample_token, detections)
    
    # 4. ai agent working
    agent = AutonomousSceneAgent()
    ai_decision = agent.analyze_scene(fusion_results)
    
    print("\n==============================================")
    print("      AI AGENT WORKING (STRUCTURED JSON)     ")
    print("==============================================")
    print(json.dumps(ai_decision, indent=4, ensure_ascii=False))