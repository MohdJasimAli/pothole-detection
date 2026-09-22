"""
Probe Render API payload schemas to create the web service, then poll the
first deploy until live and record the live URL.

Usage:
    python probe_service.py <RENDER_API_KEY> <OWNER_ID>
"""
import json
import sys
import time

from render_deploy import api_call, as_list, unwrap, REPO_URL, SERVICE_NAME

BUILD = "pip install -r requirements-render.txt && pytest tests/ -q"
START = "waitress-serve --host=0.0.0.0 --port=$PORT api.app:app"
ENVVARS = [{"key": "PYTHON_VERSION", "value": "3.12.7"},
           {"key": "API_DEBUG", "value": "false"}]

# Render's createService schema differs across API versions; try each shape,
# stop at the first success (a success CREATES the service, so order matters).
CANDIDATES = [
    # 1: repo as string, branch top-level, plan/region in serviceDetails,
    #    build/start in envSpecificDetails  (matches current docs)
    {
        "type": "web_service", "name": SERVICE_NAME, "ownerId": None,
        "repo": REPO_URL, "branch": "main", "autoDeploy": "yes",
        "serviceDetails": {
            "env": "python", "plan": "free", "region": "oregon",
            "numInstances": 1, "healthCheckPath": "/api/health",
            "envSpecificDetails": {"buildCommand": BUILD,
                                   "startCommand": START},
        },
    },
    # 2: same as 1 + envVars inside serviceDetails
    {
        "type": "web_service", "name": SERVICE_NAME, "ownerId": None,
        "repo": REPO_URL, "branch": "main", "autoDeploy": "yes",
        "serviceDetails": {
            "env": "python", "plan": "free", "region": "oregon",
            "numInstances": 1, "healthCheckPath": "/api/health",
            "envSpecificDetails": {"buildCommand": BUILD,
                                   "startCommand": START},
            "envVars": ENVVARS,
        },
    },
    # 3: plan/region at root, no healthCheck/numInstances
    {
        "type": "web_service", "name": SERVICE_NAME, "ownerId": None,
        "repo": REPO_URL, "branch": "main", "autoDeploy": "yes",
        "region": "oregon", "plan": "free",
        "serviceDetails": {
            "env": "python",
            "envSpecificDetails": {"buildCommand": BUILD,
                                   "startCommand": START},
        },
    },
    # 4: repo as git object
    {
        "type": "web_service", "name": SERVICE_NAME, "ownerId": None,
        "repo": {"type": "git", "url": REPO_URL, "branch": "main"},
        "autoDeploy": "yes",
        "serviceDetails": {
            "env": "python", "plan": "free", "region": "oregon",
            "envSpecificDetails": {"buildCommand": BUILD,
                                   "startCommand": START},
        },
    },
]


def main():
    key = sys.argv[1]
    owner_id = sys.argv[2]
    print("owner:", owner_id)

    status, svc = None, {}
    service_id = None
    for i, body in enumerate(CANDIDATES, 1):
        body["ownerId"] = owner_id
        status, svc = api_call("POST", "/services", key, body)
        msg = json.dumps(svc)[:260]
        print(f"[create] variant {i}: HTTP {status} {msg}")
        if status in (200, 201):
            service = unwrap(svc, "service")
            service_id = service.get("id")
            print("[create] SUCCESS id:", service_id)
            print("[create] dashboard:",
                  service.get("serviceDetails", {}).get("dashboardUrl", ""))
            break
        low = json.dumps(svc).lower()
        if "env" in low and ("environment" in low or "envid" in low):
            st, envs = api_call(
                "GET", f"/environments?ownerId={owner_id}", key)
            env_list = as_list(envs, "environments")
            print("[create] envId required; available:", len(env_list))
            if env_list:
                body["envId"] = env_list[0].get("id")
                print("[create] retry with envId", body["envId"])
                status, svc = api_call("POST", "/services", key, body)
                print(f"[create] envId retry: HTTP {status}",
                      json.dumps(svc)[:260])
                if status in (200, 201):
                    service_id = unwrap(svc, "service").get("id")
                    break

    if not service_id:
        print("[create] FAILED all variants")
        sys.exit(1)

    # best-effort env vars (separate endpoint in this API version)
    for method in ("PUT", "POST"):
        st, resp = api_call(method, f"/services/{service_id}/env-vars",
                            key, ENVVARS)
        print(f"[envvars] {method} -> HTTP {st}")
        if st in (200, 201):
            break

    # poll deploy to live
    print("[deploy] waiting for build+tests+start (max ~14 min) ...")
    last = ""
    for _ in range(84):
        st, deploys = api_call(
            "GET", f"/services/{service_id}/deploys?limit=1", key)
        dlist = as_list(deploys, "deploys")
        dep = dlist[0] if dlist else {}
        state = dep.get("status", "?")
        if state != last:
            print("  status:", state)
            last = state
        if state == "live":
            break
        if state in ("build_failed", "update_failed", "canceled",
                     "deactivated"):
            print("[deploy] FAILED:", state)
            sys.exit(1)
        time.sleep(10)
    else:
        print("[deploy] TIMEOUT")
        sys.exit(1)

    st, svc2 = api_call("GET", f"/services/{service_id}", key)
    s = unwrap(svc2, "service")
    url = s.get("serviceDetails", {}).get("url", "")
    print("=" * 60)
    print("DEPLOYED:", url)
    print("serviceId:", service_id)
    print("=" * 60)
    with open("render_deploy_info.json", "w") as f:
        json.dump({"url": url, "serviceId": service_id,
                   "ownerId": owner_id, "name": SERVICE_NAME}, f, indent=2)
    print("saved render_deploy_info.json")


if __name__ == "__main__":
    main()