"""Production smoke test: POST /api/analyze to the running waitress server."""
import io
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8080"
IMAGE = r"E:\Project\CODE\ProjectDev\data\samples\multi_potholes.jpg"

boundary = "----ProdSmokeTest"
with open(IMAGE, "rb") as f:
    img = f.read()

body = io.BytesIO()
body.write(("--%s\r\n" % boundary).encode())
body.write(b'Content-Disposition: form-data; name="image"; '
           b'filename="multi_potholes.jpg"\r\n'
           b"Content-Type: image/jpeg\r\n\r\n")
body.write(img)
body.write(b"\r\n--%s--\r\n" % boundary.encode())

req = urllib.request.Request(
    BASE + "/api/analyze",
    data=body.getvalue(),
    headers={"Content-Type":
             "multipart/form-data; boundary=%s" % boundary})

import time
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
        dt = time.time() - t0
        print("POST /api/analyze -> HTTP", resp.status, "in %.2fs" % dt)
        rep = data.get("report", {})
        print("potholes:", rep.get("total_potholes"))
        print("summary :", rep.get("severity_summary"))
        print("backend :", rep.get("model_info"))
        for i, p in enumerate(rep.get("potholes", []), 1):
            print("  #%d conf=%.4f size=%.1fcm2 (%s) depth=%.2fcm (%s) "
                  "severity=%s priority=%s"
                  % (i, p["confidence"], p["size_cm2"], p["size_category"],
                     p["depth_cm"], p["depth_category"], p["severity"],
                     p["priority"]))
        print("annotated base64 chars:", len(data.get(
            "annotated_image_base64", "")))
        print("PRODUCTION SMOKE TEST: PASS")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:500])
    print("PRODUCTION SMOKE TEST: FAIL")
except Exception as e:
    print("ERROR:", e)
    print("PRODUCTION SMOKE TEST: FAIL")