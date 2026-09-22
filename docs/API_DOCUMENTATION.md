# API Documentation

**Road Pothole Detection & Severity Analysis System — REST API v1.0**

Base URL: `http://localhost:5000`

---

## Quick Start

```bash
# Start the server
python main.py --web
# or
python api/app.py

# Open the dashboard
# http://localhost:5000/dashboard
```

---

## Endpoints

### 1. `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Pothole Detection & Severity Analysis API",
  "version": "1.0.0",
  "endpoints": {
    "/api/health": "GET - Health check",
    "/api/detect": "POST - Upload image, get detection results",
    "/api/analyze": "POST - Full analysis",
    "/dashboard": "GET - Web dashboard",
    "/api/reports": "GET - List all reports"
  }
}
```

---

### 2. `GET /api/health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000000",
  "device": "cpu"
}
```

---

### 3. `POST /api/detect`
Detect potholes in an uploaded image and save a report.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Road image (jpg, png, bmp, tiff) |
| `gps_lat` | float | No | GPS latitude |
| `gps_lng` | float | No | GPS longitude |

**Example (curl):**
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@road.jpg" \
  -F "gps_lat=37.7749" \
  -F "gps_lng=-122.4194"
```

**Example (Python):**
```python
import requests

with open("road.jpg", "rb") as f:
    files = {"image": f}
    data = {"gps_lat": 37.7749, "gps_lng": -122.4194}
    response = requests.post("http://localhost:5000/api/detect",
                            files=files, data=data)

report = response.json()
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "gps_coordinates": {"lat": 37.7749, "lng": -122.4194},
  "total_potholes": 3,
  "severity_summary": {
    "critical": 1,
    "high": 1,
    "medium": 1,
    "low": 0
  },
  "potholes": [
    {
      "bbox": [120, 340, 280, 460],
      "confidence": 0.9412,
      "class_id": 0,
      "class_name": "Pothole",
      "width_px": 160,
      "height_px": 120,
      "center": [200, 400],
      "size_cm2": 2457.6,
      "size_category": "Large",
      "width_cm": 64.0,
      "height_cm": 48.0,
      "depth_cm": 6.2,
      "depth_category": "Deep",
      "severity": "Critical",
      "priority": 1,
      "response_time_hours": 24
    }
  ],
  "processing_time_seconds": 1.847,
  "model_info": {
    "detector": "YOLOv8",
    "depth_model": "MiDaS DPT_Hybrid",
    "device": "cpu"
  },
  "annotated_image_path": "frontend/uploads/road_a1b2c3d4_annotated.jpg",
  "report_path": "frontend/uploads/road_a1b2c3d4_report.json",
  "report_id": "a1b2c3d4"
}
```

---

### 4. `POST /api/analyze`
Full analysis with base64-encoded annotated image for immediate display.

**Request:** Same as `/api/detect`

**Response (additional fields):**
```json
{
  "status": "success",
  "timestamp": "2024-01-15T10:30:00.000000",
  "image_info": {
    "height": 720,
    "width": 1280
  },
  "detections": [ "same as potholes above" ],
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "report": "full report object",
  "processing_time_seconds": 1.847
}
```

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const response = await fetch("/api/analyze", {
  method: "POST",
  body: formData
});

const data = await response.json();
document.getElementById("result").src =
  "data:image/jpeg;base64," + data.annotated_image_base64;
```

---

### 5. `GET /api/reports`
List all saved analysis reports.

**Response:**
```json
{
  "reports": [
    {
      "report_id": "a1b2c3d4",
      "timestamp": "2024-01-15T10:30:00",
      "total_potholes": 3,
      "severity_summary": {},
      "report_file": "road_a1b2c3d4_report.json",
      "annotated_image_file": "road_a1b2c3d4_annotated.jpg"
    }
  ],
  "count": 1
}
```

---

### 6. `GET /api/report/<report_id>`
Retrieve a specific report by ID.

```bash
curl http://localhost:5000/api/report/a1b2c3d4
```

---

### 7. `POST /api/visualize`
Get annotated image without saving a full report.

**Response:**
```json
{
  "status": "success",
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "detections": []
}
```

---

### 8. `GET /dashboard`
Serves the interactive web dashboard (HTML).

---

## Error Handling

| Status Code | Meaning | Response |
|-------------|---------|----------|
| 200 | Success | Result object |
| 400 | Bad Request | `{"error": "No image file provided"}` |
| 404 | Not Found | `{"error": "Endpoint not found"}` |
| 413 | Payload Too Large | `{"error": "File too large (max 50MB)"}` |
| 500 | Server Error | `{"error": "<message>"}` |

---

## Data Model Reference

### Detection Object

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `[int, int, int, int]` | Bounding box `[x1, y1, x2, y2]` in pixels |
| `confidence` | float | Detection confidence (0-1) |
| `class_id` | int | Class index (0 = Pothole) |
| `class_name` | string | Class label |
| `width_px` | int | Bounding box width in pixels |
| `height_px` | int | Bounding box height in pixels |
| `center` | `[int, int]` | Center point `[cx, cy]` |
| `size_cm2` | float | Estimated area in cm2 |
| `size_category` | string | `Small` / `Medium` / `Large` |
| `width_cm` | float | Estimated width in cm |
| `height_cm` | float | Estimated height in cm |
| `depth_cm` | float | Estimated depth in cm |
| `depth_category` | string | `Shallow` / `Moderate` / `Deep` |
| `severity` | string | `Low` / `Medium` / `High` / `Critical` |
| `priority` | int | 1 (Critical) to 4 (Low) |
| `response_time_hours` | int | Recommended response deadline |

### Severity Levels

| Severity | Priority | Response Time | Color |
|----------|----------|---------------|-------|
| Critical | 1 | 24 hours | Red |
| High | 2 | 1 week (168h) | Orange |
| Medium | 3 | 1 month (720h) | Yellow |
| Low | 4 | 3 months (2160h) | Green |

---

## Configuration

Edit `config.py` or create a `.env` file:

```bash
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True
CONFIDENCE_THRESHOLD=0.4
MAX_CONTENT_LENGTH_MB=50
```

---

## Limits

- **Max file size:** 50 MB (configurable)
- **Supported formats:** JPG, JPEG, PNG, BMP, TIFF, GIF
- **Concurrent requests:** Limited by available CPU/GPU
- **Processing time:** ~1-2s per image on CPU, ~0.2s on GPU

---

## Integration Examples

### Python Client

```python
import requests

class PotholeDetectorClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def analyze(self, image_path, gps=None):
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {}
            if gps:
                data["gps_lat"], data["gps_lng"] = gps
            r = requests.post(f"{self.base_url}/api/analyze",
                             files=files, data=data)
        return r.json()

client = PotholeDetectorClient()
result = client.analyze("road.jpg", gps=(37.7749, -122.4194))
print(f"Found {result['report']['total_potholes']} potholes")
```

### JavaScript Fetch

```javascript
async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  const res = await fetch('/api/analyze', {
    method: 'POST',
    body: formData
  });

  return await res.json();
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | Install: `pip install torch ultralytics` |
| Model download fails | Manually place `best.pt` in `data/models/` |
| Slow inference | Use GPU or enable YOLOv8n (nano) model |
| Out of memory | Reduce `IMAGE_SIZE` in config.py |
| File too large | Increase `MAX_CONTENT_LENGTH_MB` in config.py |

---

**API Version:** 1.0.0
**Last Updated:** 2024
**Authors:** ProjectDev Team