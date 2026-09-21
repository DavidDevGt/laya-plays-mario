"""Serve the Laya-plays-Mario demo locally.

Usage:
    python run_demo.py                # record if needed, then serve on http://localhost:8731
    python run_demo.py --record       # force a fresh recording even if episode.* already exists
    python run_demo.py --port 8080

Everything the page shows is read from episode.jsonl / episode.mp4 / episode_meta.json, produced
by record_demo.py actually running the real mario_laya_v6_dagger checkpoint through the real
controller loop -- this script only records (if needed) and serves those files; it does not
generate or alter any of the numbers the UI displays.
"""
import argparse
import http.server
import mimetypes
import os
import re
import subprocess
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class DemoServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True  # avoid "Address already in use" on quick restarts


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler doesn't support HTTP Range requests, which browsers rely on to
    seek in <video>: without a 206 Partial Content response, scrubbing the timeline doesn't work
    reliably. This adds real Range support for GET requests; everything else (index.html, the
    JSONL log, static assets) falls back to the normal full-file behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/static/index.html"
        range_header = self.headers.get("Range")
        if range_header:
            served = self._serve_range(range_header)
            if served:
                return
        return super().do_GET()

    def _serve_range(self, range_header):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return False
        m = RANGE_RE.match(range_header)
        if not m:
            return False
        file_size = os.path.getsize(path)
        start_s, end_s = m.groups()
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return True
        length = end - start + 1
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk = 64 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(data)
        return True

    def end_headers(self):
        if not self.path.startswith("/episode.jsonl") and not self.path.startswith("/episode_meta.json"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; the browser console + video are the demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--record", action="store_true", help="force re-recording the episode")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    video_path = os.path.join(HERE, "episode.mp4")
    jsonl_path = os.path.join(HERE, "episode.jsonl")
    need_recording = args.record or not (os.path.exists(video_path) and os.path.exists(jsonl_path))

    if need_recording:
        print("Recording a fresh episode with the real v6_dagger checkpoint (this runs actual model inference)...")
        result = subprocess.run([sys.executable, os.path.join(HERE, "record_demo.py")])
        if result.returncode != 0:
            print("Recording failed; aborting.")
            sys.exit(1)
    else:
        print(f"Using existing recording: {video_path}")
        print("  (pass --record to generate a fresh one)")

    url = f"http://localhost:{args.port}/"
    print(f"\nServing demo at {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    with DemoServer(("0.0.0.0", args.port), DemoHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
