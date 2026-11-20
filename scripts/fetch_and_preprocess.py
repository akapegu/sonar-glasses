#!/usr/bin/env python3
"""Download the SoniSpeech zips from Cornell eCommons and build the cache, one zip at a time.

Per zip: download (resumable) -> verify size + MD5 -> unzip without video.mp4 ->
scripts/preprocess.py into the cache -> delete the zip and the extracted audio.
Only the cache (and the small official echo_profiles/ code from zip 01) is kept,
so the raw data never needs to fit on the persistent volume.

Resumable: a finished zip leaves <cache>/_fetch/<zip>.done and is skipped next time;
an interrupted download continues where it stopped; finished sessions are skipped
by preprocess.py.

Usage (on the pod, from the repo root):
  python scripts/fetch_and_preprocess.py --cache /workspace/cache/v1 --work /root/work
  python scripts/fetch_and_preprocess.py --cache ... --work ... --zips 1 2   # subset
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://ecommons.cornell.edu/server/api"
BUNDLE = "8f1edc3c-d655-43e1-bcce-5fb790adb03e"  # ORIGINAL bundle of the SoniSpeech data item


def list_zips():
    with urllib.request.urlopen(f"{API}/core/bundles/{BUNDLE}/bitstreams?size=50", timeout=60) as r:
        items = json.load(r)["_embedded"]["bitstreams"]
    zips = {}
    for b in items:
        if b["name"].endswith(".zip"):
            k = int(b["name"].split("_")[1])
            zips[k] = dict(name=b["name"], size=b["sizeBytes"], md5=b["checkSum"]["value"],
                           url=b["_links"]["content"]["href"])
    return dict(sorted(zips.items()))


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="persistent cache dir (network volume)")
    ap.add_argument("--work", required=True, help="scratch dir for downloads (container disk)")
    ap.add_argument("--zips", nargs="*", type=int, help="zip numbers (default: all 9; zip 1 always first)")
    ap.add_argument("--workers", type=int, help="passed to preprocess.py")
    args = ap.parse_args()
    cache, work = Path(args.cache), Path(args.work)
    (cache / "_fetch").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    code_dir = cache / "_official_code" / "echo_profiles"

    zips = list_zips()
    todo = sorted(set(args.zips or zips))
    if not code_dir.exists() and 1 not in todo:
        todo = [1] + todo  # zip 01 carries the official preprocessing code
    print(f"zips to process: {todo}", flush=True)

    for k in todo:
        z = zips[k]
        marker = cache / "_fetch" / f"{z['name']}.done"
        if marker.exists():
            print(f"== {z['name']}: done already, skipping", flush=True)
            continue
        t0 = time.time()
        zpath = work / z["name"]
        print(f"== {z['name']} ({z['size'] / 1e9:.2f} GB)", flush=True)
        if not (zpath.exists() and zpath.stat().st_size == z["size"]):
            run(["curl", "-fL", "--retry", "10", "--retry-delay", "5", "-C", "-", "-o", str(zpath), z["url"]])
        size = zpath.stat().st_size
        if size != z["size"]:
            sys.exit(f"size mismatch for {z['name']}: {size} != {z['size']}")
        got = md5sum(zpath)
        if got != z["md5"]:
            zpath.unlink()
            sys.exit(f"MD5 mismatch for {z['name']} (deleted, rerun to download again): {got} != {z['md5']}")
        print(f"   verified size + MD5 ({time.time() - t0:.0f} s)", flush=True)

        ext = work / z["name"].replace(".zip", "")
        shutil.rmtree(ext, ignore_errors=True)
        run(["unzip", "-q", str(zpath), "-x", "*/video.mp4", "-d", str(ext)])
        if (ext / "echo_profiles").exists() and not code_dir.exists():
            shutil.copytree(ext / "echo_profiles", code_dir)
            print(f"   saved official code to {code_dir}", flush=True)

        cmd = [sys.executable, str(ROOT / "scripts" / "preprocess.py"), "--data", str(ext),
               "--out", str(cache), "--code", str(code_dir)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        run(cmd)

        shutil.rmtree(ext)
        zpath.unlink()
        marker.write_text(json.dumps(dict(md5=z["md5"], seconds=round(time.time() - t0))))
        print(f"== {z['name']} finished in {time.time() - t0:.0f} s", flush=True)

    print("all requested zips done; manifest at", cache / "manifest.csv", flush=True)


if __name__ == "__main__":
    main()
