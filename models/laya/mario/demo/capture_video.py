"""Record the live demo page (gameplay + synced panel) as a single playable video file.

This drives a real headless Chromium against the real running demo server -- the panel updates
are the same JS reading the same episode.jsonl as an interactive session, this just captures the
pixels while the embedded episode video plays through once, start to finish.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8731/")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "videos", "laya_demo_panel.mp4"))
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--tail-seconds", type=float, default=1.5, help="extra recording after playback ends")
    args = ap.parse_args()

    tmp_dir = os.path.join(HERE, "_capture_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=tmp_dir,
            record_video_size={"width": args.width, "height": args.height},
        )
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function('document.getElementById("video").readyState >= 2', timeout=15000)
        duration = page.evaluate('document.getElementById("video").duration')
        print(f"Episode video duration: {duration:.2f}s -- recording page for {duration + args.tail_seconds:.2f}s")
        # video has autoplay+muted; make sure it's actually playing from the start
        page.evaluate('const v = document.getElementById("video"); v.currentTime = 0; v.play();')
        page.wait_for_timeout(int((duration + args.tail_seconds) * 1000))
        context.close()  # finalizes the .webm file
        browser.close()

    webm_files = glob.glob(os.path.join(tmp_dir, "*.webm"))
    if not webm_files:
        print("No recording produced.")
        sys.exit(1)
    webm_path = webm_files[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", webm_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", "30",
        args.out,
    ], check=True)
    shutil.rmtree(tmp_dir)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
