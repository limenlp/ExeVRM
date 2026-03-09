#!/usr/bin/env python3
"""Replace video path prefix in ver53k.jsonl and verbench.jsonl.

Usage:
    python replace_video_prefix.py <old_prefix> <new_prefix>

Example:
    python replace_video_prefix.py /export/home/VER_53k /data/VER_53k
    # /export/home/VER_53k/train/xxx.mp4 -> /data/VER_53k/train/xxx.mp4
"""

import argparse
import json
import os

JSONL_FILES = [
    os.path.join(os.path.dirname(__file__), "data/ver53k/ver53k.jsonl"),
    os.path.join(os.path.dirname(__file__), "data/verbench.jsonl"),
]


def replace_prefix(file_path, old_prefix, new_prefix):
    with open(file_path) as f:
        data = json.load(f)

    count = 0
    for item in data:
        new_videos = []
        for v in item.get("videos", []):
            if v.startswith(old_prefix):
                v = new_prefix + v[len(old_prefix):]
                count += 1
            new_videos.append(v)
        item["videos"] = new_videos

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return count


def main():
    parser = argparse.ArgumentParser(description="Replace video path prefix in dataset files.")
    parser.add_argument("--old_prefix", default="/export/home/VER_53k", help="Old path prefix to replace (default: /export/home/VER_53k)")
    parser.add_argument("new_prefix", help="New path prefix")
    args = parser.parse_args()

    old = args.old_prefix.rstrip("/")
    new = args.new_prefix.rstrip("/")

    if old == new:
        print("Old and new prefix are the same. Nothing to do.")
        return

    for fpath in JSONL_FILES:
        count = replace_prefix(fpath, old, new)
        print(f"{os.path.basename(fpath)}: replaced {count} paths")

    print("Done.")


if __name__ == "__main__":
    main()
