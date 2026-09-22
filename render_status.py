"""Query Render API for the status of our service(s) and latest deploy."""
import json
import sys

from render_deploy import api_call, as_list, unwrap

key = sys.argv[1]

st, body = api_call("GET", "/services?limit=20", key)
print("GET /services -> HTTP", st)
items = as_list(body, "services")
if not items:
    print("  (no services listed) raw:", json.dumps(body)[:300])

for it in items:
    svc = unwrap(it, "service")
    sid = svc.get("id")
    name = svc.get("name")
    stype = svc.get("type")
    sd = svc.get("serviceDetails", {}) or {}
    url = sd.get("url", "")
    print(f"\nSERVICE {name} [{stype}] id={sid}")
    print("  url:", url)
    print("  suspended:", svc.get("suspended"), "autoDeploy:", svc.get("autoDeploy"))
    # latest deploy
    st2, dep_body = api_call("GET", f"/services/{sid}/deploys?limit=1", key)
    dlist = as_list(dep_body, "deploys")
    if dlist:
        d = unwrap(dlist[0], "deploy")
        print("  latest deploy:", d.get("id"), "status:", d.get("status"))
        if d.get("status") in ("build_failed", "update_failed"):
            st3, logs = api_call(
                "GET",
                f"/services/{sid}/deploys/{d.get('id')}/events?limit=30", key)
            for ev in as_list(logs, "events"):
                e = unwrap(ev, "event")
                if e.get("type") == "build_failed" or "error" in str(
                        e.get("type", "")).lower():
                    print("   EVENT:", json.dumps(e)[:300])
    else:
        print("  no deploys yet")