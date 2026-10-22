import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from nuscenes.nuscenes import NuScenes

from dataset import NuScenesDataset
from detector import SceneDetector
from fusion import calculate_distances
from agent import AutonomousSceneAgent

BASE_DIR = Path(__file__).resolve().parent
DATAROOT = BASE_DIR / "data" / "nuscenes"

state = {}


@asynccontextmanager
async def lifespan(app):
    state["nusc"] = NuScenes(version="v1.0-mini", dataroot=str(DATAROOT), verbose=False)
    state["detector"] = SceneDetector()
    state["agent"] = AutonomousSceneAgent()
    yield
    state.clear()


app = FastAPI(title="Agentic Autonomous Scene Analyzer", lifespan=lifespan)


def get_dataset(scene: int, frame: int) -> NuScenesDataset:
    nusc = state["nusc"]
    if scene >= len(nusc.scene):
        raise HTTPException(status_code=404, detail=f"Scene {scene} not found")

    dataset = NuScenesDataset(nusc, scene_index=scene)
    if frame >= len(dataset):
        raise HTTPException(status_code=404, detail=f"Frame {frame} not found in scene {scene}")
    return dataset


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scenes")
def list_scenes():
    return [
        {"index": i, "name": scene["name"], "description": scene["description"]}
        for i, scene in enumerate(state["nusc"].scene)
    ]


@app.get("/frame")
def frame_image(scene: int = Query(0, ge=0), frame: int = Query(0, ge=0)):
    dataset = get_dataset(scene, frame)
    buffer = io.BytesIO()
    dataset[frame].save(buffer, format="JPEG")
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


@app.post("/analyze")
def analyze(scene: int = Query(0, ge=0), frame: int = Query(0, ge=0)):
    dataset = get_dataset(scene, frame)
    image = dataset[frame]
    detections = state["detector"].detect(image)
    fusion_results = calculate_distances(state["nusc"], dataset.sample_tokens[frame], detections)
    decision = state["agent"].analyze_scene(fusion_results)

    return {
        "scene": scene,
        "frame": frame,
        "detections": fusion_results,
        "decision": decision,
    }
