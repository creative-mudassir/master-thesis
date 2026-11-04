"""Stall test for the GDAL network settings (edc.cloudnative.GDAL_NET).

    python tests/test_stall_gdal.py

A tiny local web server pretends to be a COG server: it answers the size request normally,
but every data request gets its headers and then NO data, like the frozen download seen on
2026-11-03. With GDAL_NET, GDAL must give up with an error instead of waiting forever.
Takes a few minutes (60 s stall limit per try, plus GDAL's own retries).
"""
import http.server
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rasterio  # noqa: E402

from edc.cloudnative import GDAL_NET  # noqa: E402

SIZE = 10_000_000
GIVE_UP_AFTER = 20 * 60    # the test fails if GDAL is still waiting after 20 min (= the watchdog)
requests_seen = []


class FrozenCOG(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        requests_seen.append(("HEAD", time.time()))
        self.send_response(200)
        self.send_header("Content-Length", str(SIZE))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        requests_seen.append(("GET", time.time()))
        self.send_response(206)
        self.send_header("Content-Range", f"bytes 0-16383/{SIZE}")
        self.send_header("Content-Length", "16384")
        self.end_headers()
        self.wfile.flush()
        time.sleep(GIVE_UP_AFTER + 60)    # ... and then nothing: a frozen download

    def log_message(self, *a):
        pass


def test_stalled_download_becomes_error():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FrozenCOG)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"/vsicurl/http://127.0.0.1:{srv.server_port}/frozen.tif"
    result = {}

    def read():
        t0 = time.time()
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", **GDAL_NET):
                rasterio.open(url)
            result["outcome"] = "opened (unexpected)"
        except Exception as e:
            result["outcome"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        result["seconds"] = round(time.time() - t0, 1)

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(GIVE_UP_AFTER)
    gets = sum(1 for r in requests_seen if r[0] == "GET")
    print(f"GDAL {rasterio.__gdal_version__}; settings {GDAL_NET}")
    print(f"result: {result or 'STILL WAITING after %d s' % GIVE_UP_AFTER}; data requests sent: {gets}")
    assert result.get("outcome", "").startswith("error"), "stalled download did not become an error"
    return result


if __name__ == "__main__":
    test_stalled_download_becomes_error()
    print("stall test passed")
