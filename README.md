# Agentic Autonomous Scene Analyzer

An end-to-end pipeline that analyzes a driving scene from a front camera and a lidar sensor, detects and localizes objects, and asks an LLM agent for a structured safety decision. Results are served through a FastAPI web application that runs under Uvicorn and can be deployed with Docker.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed.svg)

## Features

- **Object detection** with YOLOv8 on the nuScenes front camera (`CAM_FRONT`).
- **Distance estimation** by projecting the `LIDAR_TOP` point cloud onto the camera image and taking the median depth inside each detection box.
- **LLM decision making** with a Groq-hosted model. It returns a risk level, the most critical object, a recommended action, and the reasoning, in strict JSON.
- **Web interface** to pick a scene and frame, view the image with detections drawn on it, and read the decision.
- **REST API** for integration with other tools.
- **Docker support** for reproducible deployment.

## Project Structure

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI application: web page, API routes, and model loading at startup |
| `static/index.html` | Single-page web interface (HTML, CSS, and JavaScript) |
| `main.py` | Runs the full pipeline once from the command line and saves the results to `outputs/` |
| `dataset.py` | Loads front camera frames from a nuScenes scene |
| `detector.py` | YOLOv8 object detection wrapper |
| `fusion.py` | Lidar-to-camera projection and per-object distance estimation |
| `agent.py` | LLM agent that produces the structured safety decision |
| `hazard_model.py` | Prototype PyTorch MLP that scores hazard per object (standalone, not used by the server) |
| `tracking.py` | Multi-frame distance tracker that estimates approach speed (standalone, not used by the server) |
| `Dockerfile` | Container image definition |
| `.dockerignore` | Keeps data, secrets, and local environments out of the image |

## Requirements

- Python 3.10 or newer
- A [Groq API key](https://console.groq.com/)
- The [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes#download) dataset
- Docker (optional, for containerized deployment)

## Setup

### 1. Get the dataset

Download the nuScenes v1.0-mini split and extract it so the folder layout looks like this:

```
data/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
```

The `data/` folder is excluded from Git and from the Docker image.

### 2. Configure the API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Keep this file private. It is excluded from Git and from the Docker image.

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

### Locally

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The server loads the dataset, the YOLOv8 model, and the agent once at startup, so the first start takes a few seconds.

### With Docker

```bash
docker build -t aasa .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" aasa
```

The `data/` folder is mounted into the container at runtime rather than copied into the image, which keeps the image small.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check, returns `{"status": "ok"}` |
| `GET` | `/scenes` | Lists the available scenes with index, name, and description |
| `GET` | `/frame?scene={n}&frame={m}` | Returns the camera image for a frame as JPEG |
| `POST` | `/analyze?scene={n}&frame={m}` | Runs detection, distance estimation, and the LLM agent on a frame |

Interactive API documentation is available at `/docs` while the server is running.

### Example response from `POST /analyze`

```json
{
  "scene": 0,
  "frame": 0,
  "detections": [
    {
      "label": "truck",
      "confidence": 0.89,
      "box": [122.8, 225.8, 601.8, 628.7],
      "distance_m": 12.3
    }
  ],
  "decision": {
    "scene_summary": "The ego-vehicle is approaching a truck at 12.3 meters.",
    "risk_level": "MEDIUM",
    "critical_object": "TRUCK at 12.3 meters",
    "recommended_action": "SLOW_DOWN",
    "reasoning": "The truck is the closest high-confidence object."
  }
}
```

Possible values:

- `risk_level`: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- `recommended_action`: `MAINTAIN_SPEED`, `SLOW_DOWN`, `BRAKE_IMMEDIATELY`, `YIELD`

## Running the Pipeline from the Command Line

To run one frame through the full pipeline and save the results to `outputs/`:

```bash
python main.py
```

This writes `outputs/detected_scene.jpg` and `outputs/agent_decision.json`.

## Security Notes

- The `/analyze` endpoint calls a paid LLM API on every request, and the application has no authentication. Add authentication and rate limiting before exposing it to the internet.
- Never commit `.env` or share your API key.

## License

No license has been specified yet.
