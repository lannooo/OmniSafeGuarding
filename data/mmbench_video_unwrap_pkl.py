import os
import pickle
import sys


def unwrap_hf_pkl(root_dir: str, suffix: str = ".mp4") -> int:
    base_dir = os.path.join(root_dir, "video_pkl")
    target_dir = os.path.join(root_dir, "video")

    if not os.path.isdir(base_dir):
        print(f"[ERROR] video_pkl directory not found: {base_dir}")
        return 1

    pickle_files = [
        os.path.join(base_dir, file)
        for file in os.listdir(base_dir)
        if file.endswith(".pkl")
    ]
    pickle_files.sort()

    if not pickle_files:
        print(f"[WARN] No .pkl files found under {base_dir}")
        return 0

    os.makedirs(target_dir, exist_ok=True)

    restored = 0
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as file:
            video_data = pickle.load(file)
        for video_name, video_content in video_data.items():
            output_path = os.path.join(target_dir, f"{video_name}{suffix}")
            with open(output_path, "wb") as output_file:
                output_file.write(video_content)
            restored += 1

    print(f"[INFO] Restored {restored} videos from pickle files into {target_dir}")
    return 0


def main() -> int:
    root_dir = None
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    elif os.environ.get("MMBENCH_VIDEO_ROOT"):
        root_dir = os.environ["MMBENCH_VIDEO_ROOT"]
    else:
        root_dir = "./data/MMBench-Video"

    return unwrap_hf_pkl(root_dir)


if __name__ == "__main__":
    raise SystemExit(main())