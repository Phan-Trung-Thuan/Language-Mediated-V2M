import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import CLIPProcessor, CLIPModel
from moviepy.editor import VideoFileClip, concatenate_videoclips

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CLUSTER_DURATION = 50.0

def sample_frames_from_video(video_path: str, n_frames: int = 6):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if frame_count <= 0:
        cap.release()
        return frames
    indices = np.linspace(0, frame_count - 1, num=min(n_frames, frame_count), dtype=int)
    idx_set = set(indices.tolist())
    cur_index = 0
    ret, frame = cap.read()
    while ret and idx_set:
        if cur_index in idx_set:
            frames.append(frame.copy())
            idx_set.remove(cur_index)
        cur_index += 1
        ret, frame = cap.read()
    cap.release()
    return frames

def embed_scene_frames(frames: List[np.ndarray], model, processor, device=DEVICE):
    if len(frames) == 0:
        return np.zeros(model.config.projection_dim if hasattr(model.config, "projection_dim") else 1024, dtype=np.float32)
    imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).cpu()
    feat = image_features.float().mean(dim=0)
    feat = feat / (feat.norm(p=2) + 1e-10)
    return feat.numpy()

def build_scene_embeddings(manifest: List[Dict[str,Any]], model, processor, n_frames=6):
    out = []
    for s in tqdm(manifest, desc="Embedding scenes"):
        frames = sample_frames_from_video(s["scene_path"], n_frames=n_frames)
        emb = embed_scene_frames(frames, model, processor)
        out.append({
            "scene_index": s["scene_index"],
            "start_seconds": s["start_seconds"],
            "end_seconds": s["end_seconds"],
            "duration_seconds": s["duration_seconds"],
            "scene_path": s["scene_path"],
            "embedding": emb
        })
    return out

def cosine_distance(a, b):
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def conditional_adjacency_clustering(scenes_embedded, max_duration=MAX_CLUSTER_DURATION):
    distances = [cosine_distance(scenes_embedded[i]["embedding"], scenes_embedded[i+1]["embedding"]) for i in range(len(scenes_embedded)-1)]
    clusters = [{
        "scenes": [item["scene_index"]],
        "start": item["start_seconds"],
        "end": item["end_seconds"],
        "duration": item["duration_seconds"],
        "path": [item["scene_path"]],
        "center": item["embedding"]
    } for item in scenes_embedded]
    while np.min(distances) < 100.0 and len(clusters) > 1:
        min_idx = np.argmin(distances)
        if clusters[min_idx]["duration"] + clusters[min_idx + 1]["duration"] <= MAX_CLUSTER_DURATION:
            tmp = clusters[min_idx]["center"] * clusters[min_idx]["duration"] + clusters[min_idx + 1]["center"] * clusters[min_idx + 1]["duration"]
            clusters[min_idx]["center"] = tmp / (clusters[min_idx]["duration"] + clusters[min_idx + 1]["duration"])
            clusters[min_idx]["scenes"].extend(clusters[min_idx + 1]["scenes"])
            clusters[min_idx]["end"] = clusters[min_idx + 1]["end"]
            clusters[min_idx]["duration"] += clusters[min_idx + 1]["duration"]
            clusters[min_idx]["path"].extend(clusters[min_idx + 1]["path"])
            del clusters[min_idx + 1]
            del distances[min_idx]
            distances[min_idx-1] = cosine_distance(clusters[min_idx - 1]["center"], clusters[min_idx]["center"])
            if min_idx < len(distances) - 1:
                distances[min_idx] = cosine_distance(clusters[min_idx]["center"], clusters[min_idx + 1]["center"])
        else:
            distances[min_idx] = 100.0
    return clusters

def merge_cluster(cluster, output_path):
    clips = [VideoFileClip(path) for path in cluster["path"]]
    merged = concatenate_videoclips(clips, method="compose")
    merged.write_videofile(output_path, codec="libx264", audio_codec="aac")
    for c in clips:
        c.close()
    merged.close()
    return output_path
