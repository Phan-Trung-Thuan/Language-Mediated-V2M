import cv2
import json
from pathlib import Path
from typing import List, Tuple
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes_py(video_path: str, threshold: float = 27.0, downscale_factor: int = 2) -> List[Tuple[float, float]]:
    vm = VideoManager([str(video_path)])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    vm.set_downscale_factor(downscale_factor)
    try:
        vm.start()
        sm.detect_scenes(frame_source=vm)
        scene_list = sm.get_scene_list()
        scenes = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
        return scenes
    finally:
        vm.release()

def reencode_segment_with_opencv(input_path, start_s, end_s, output_path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_id += 1
    cap.release()
    out.release()

def split_video_into_scenes_and_save(video_path: str, output_dir: str, detector_threshold: float = 16.0, downscale_factor: int = 2, container_ext: str = "mp4") -> str:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    scenes_dir = outdir / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    print('Detecting scenes...')
    scenes = detect_scenes_py(str(video_path), threshold=detector_threshold, downscale_factor=downscale_factor)
    if not scenes:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            dur = float(frames / fps) if frames > 0 else 0.0
        finally:
            cap.release()
        scenes = [(0.0, dur)]
    manifest = []
    base_stem = video_path.stem
    for idx, (start_s, end_s) in enumerate(scenes, start=1):
        duration = max(0.0, end_s - start_s)
        out_name = f"{base_stem}_scene{idx:03d}.{container_ext}"
        out_file = scenes_dir / out_name
        reencode_segment_with_opencv(video_path, start_s, end_s, out_file)
        manifest.append({
            "scene_index": idx,
            "start_seconds": float(start_s),
            "end_seconds": float(end_s),
            "duration_seconds": float(duration),
            "scene_path": str(out_file.resolve())
        })
    manifest_path = outdir / f"{base_stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return str(manifest_path)
