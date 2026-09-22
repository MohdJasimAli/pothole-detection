"""
Deploy the project to Render.com via the Render REST API.

Usage:
    python render_deploy.py <RENDER_API_KEY>

What it does:
    1. Validates the API key (GET /v1/owners)
    2. Creates a Python web service "pothole-detection-api" from the
       PUBLIC GitHub repo https://github.com/MohdJasimAli/pothole-detection.git
       (buildCommand runs requirements install + pytest, startCommand runs
       waitress exactly as specified in render.yaml)
    3. Polls the first deploy until status == "live" (up to 15 min)
    4. Prints the live URL; then run prod_smoke.py against it

No third-party SDKs - pure stdlib urllib, so it runs on the existing env.
"""
import json
import sys
import time
import urllib.request
import urllib.error

API = "https://api.render.com/v1"
REPO_URL = "https://github.com/MohdJasimAli/pothole-detection.git"
SERVICE_NAME = "pothole-detection-api"
OWNER_NAME = "MohdJasimAli"


def api_call(method, path, api_key, body=None):
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:800]}


def as_list(body, key):
    """Render responses vary: list, {key: [...]}, or a single dict."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        val = body.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [val]
        return [body]
    return []


def unwrap(body, key):
    """Return the main object from an API response body."""
    if isinstance(body, list):
        return body[0] if body else {}
    if isinstance(body, dict) and isinstance(body.get(key), dict):
        return body[key]
    return body if isinstance(body, dict) else {}


def main():
    if len(sys.argv) < 2:
        print("usage: python render_deploy.py <RENDER_API_KEY>")
        sys.exit(2)
    key = sys.argv[1]

    # 1. Validate key + get owner id
    print("[1/4] Validating Render API key ...")
    status, owners_body = api_call("GET", "/owners", key)
    owner_list = as_list(owners_body, "owners")
    # paginated shape: [{"cursor": "...", "owner": {...}}]
    if owner_list and isinstance(owner_list[0].get("owner"), dict):
        owner_list = [it["owner"] for it in owner_list]
    if status != 200 or not owner_list or "id" not in owner_list[0]:
        print("  FAIL: HTTP", status, json.dumps(owners_body)[:500])
        print("  -> check the key (Dashboard > Account Settings > API Keys)")
        sys.exit(1)
    owner = owner_list[0]
    owner_id = owner["id"]
    print("  OK owner:", owner.get("name") or owner.get("email"), owner_id)

    # 2. Create the web service from the public repo
    print("[2/4] Creating web service from GitHub repo ...")
    service_body = {
        "type": "web_service",
        "name": SERVICE_NAME,
        "ownerId": owner_id,
        "plan": "free",
        "region": "oregon",
        "repo": {
            "type": "git",
            "url": REPO_URL,
            "branch": "main",
        },
        "serviceDetails": {
            "runtime": "python",
            "buildCommand": "pip install -r requirements-render.txt && pytest tests/ -q",
            "startCommand":
                "waitress-serve --host=0.0.0.0 --port=$PORT api.app:app",
            "numInstances": 1,
            "healthCheckPath": "/api/health",
            "autoDeploy": "yes",
            "envVars": [
                {"key": "PYTHON_VERSION", "value": "3.12.7"},
                {"key": "API_DEBUG", "value": "false"},
                {"key": "FLASK_SECRET_KEY", "generateValue": True},
            ],
        },
    }
    status, svc = api_call("POST", "/services", key, service_body)

    # Fallback: some accounts require an environment id
    if status in (400, 422) and "env" in json.dumps(svc).lower():
        print("  envId required - fetching environments ...")
        st, envs = api_call(
            "GET", f"/environments?ownerId={owner_id}", key)
        env_list = as_list(envs, "environments")
        if env_list:
            service_body["envId"] = env_list[0]["id"]
            print("  using env:", env_list[0]["id"])
            status, svc = api_call("POST", "/services", key, service_body)

    if status not in (200, 201):
        print("  FAIL: HTTP", status, json.dumps(svc)[:800])
        sys.exit(1)
    service = unwrap(svc, "service")
    if "id" not in service:
        print("  FAIL: no service id in response:", json.dumps(svc)[:800])
        sys.exit(1)
    service_id = service["id"]
    print("  OK service:", service_id, service.get("name"))
    print("  dashboard:", service.get("serviceDetails", {}).get("dashboardUrl", ""))

    # 3. Poll first deploy until live (or failed)
    print("[3/4] Waiting for first deploy (build + tests + start), up to 15 min ...")
    last = ""
    for _ in range(90):
        st, deploys = api_call(
            "GET", f"/services/{service_id}/deploys?limit=1", key)
        deploy_list = as_list(deploys, "deploys")
        deploy = deploy_list[0] if deploy_list else {}
        state = deploy.get("status", "?")
        if state != last:
            print("   deploy status:", state)
            last = state
        if state == "live":
            break
        if state in ("build_failed", "deleted", "deactivated",
                     "canceled", "update_failed"):
            print("  FAIL:", state)
            print("  build log endpoint:",
                  f"/services/{service_id}/deploys/{deploy.get('id')}/logs")
            sys.exit(1)
        time.sleep(10)
    else:
        print("  TIMEOUT waiting for deploy")
        sys.exit(1)

    # 4. Get live URL
    print("[4/4] Fetching live URL ...")
    st, svc2 = api_call("GET", f"/services/{service_id}", key)
    s = unwrap(svc2, "service")
    url = s.get("serviceDetails", {}).get("url") or \
        s.get("serviceDetails", {}).get("serviceUrl")
    print("=" * 60)
    print("DEPLOYED:", url)
    print("serviceId:", service_id)
    print("=" * 60)
    # persist for smoke test
    with open("render_deploy_info.json", "w") as f:
        json.dump({"url": url, "serviceId": service_id,
                   "ownerId": owner_id, "name": SERVICE_NAME}, f, indent=2)
    print("saved render_deploy_info.json")


if __name__ == "__main__":
    main()