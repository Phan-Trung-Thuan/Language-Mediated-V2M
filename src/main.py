import os
import json
import time

from scene_segmentation import split_video_into_scenes_and_save
from video_clustering import build_scene_embeddings, conditional_adjacency_clustering, merge_cluster, MAX_CLUSTER_DURATION
from video_captioning import get_captioning_model, caption_clusters
from music_feature_suggestion import get_qwen_model, suggest_music_features_and_prompt
from music_generation import generate_music, normalize_audio, build_final_video_from_dir

def main(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scene Segmentation
    print("Stage 1: Scene Segmentation")
    manifest_path = split_video_into_scenes_and_save(video_path, output_dir)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # 2. Video Clustering
    print("Stage 2: Video Clustering")
    # Assuming CLIP models are loaded here or inside the functions
    # (In practice, you'd pass the model and processor down)
    from transformers import CLIPProcessor, CLIPModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    
    scenes_embedded = build_scene_embeddings(manifest, clip_model, clip_processor)
    clusters = conditional_adjacency_clustering(scenes_embedded, max_duration=MAX_CLUSTER_DURATION)
    
    cluster_dir = os.path.join(output_dir, "clusters")
    os.makedirs(cluster_dir, exist_ok=True)
    
    for i, cluster in enumerate(clusters):
        merged_path = merge_cluster(cluster, os.path.join(cluster_dir, f"cluster_{i+1}.mp4"))
        clusters[i]['path'] = merged_path
        if 'center' in clusters[i]:
            del clusters[i]['center']
            
    clusters_path = os.path.join(output_dir, "clusters_info.json")
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
        
    del clip_model
    torch.cuda.empty_cache()
    
    # 3. Video Captioning
    print("Stage 3: Video Captioning")
    cap_model, cap_processor = get_captioning_model()
    caption_clusters(clusters_path, cap_model, cap_processor, clusters_path)
    del cap_model
    torch.cuda.empty_cache()
    
    # 4. Music Feature Suggestion
    print("Stage 4: Music Feature Suggestion")
    qwen_model, qwen_tokenizer = get_qwen_model()
    suggest_music_features_and_prompt(clusters_path, qwen_model, qwen_tokenizer, clusters_path)
    del qwen_model
    torch.cuda.empty_cache()
    
    # 5. Music Generation
    print("Stage 5: Music Generation")
    from audiocraft.models import MusicGen
    musicgen_model = MusicGen.get_pretrained("facebook/musicgen-medium", device=device)
    musicgen_model.set_generation_params(use_sampling=True, top_k=250, duration=30)
    
    music_dir = os.path.join(output_dir, "music")
    os.makedirs(music_dir, exist_ok=True)
    
    with open(clusters_path, 'r', encoding="utf-8") as f:
        clusters = json.load(f)
        
    prev_audio_path = None
    for i, cluster in enumerate(clusters, start=1):
        prompt = cluster["musicgen_prompt"]
        duration = cluster["duration"]
        audio_path = os.path.join(music_dir, f"music_cluster_{i}.wav")
        generate_music(musicgen_model, prompt, duration, audio_path, prev_audio_path, device=device)
        normalize_audio(audio_path)
        prev_audio_path = audio_path
        
    # 6. Build Final Video
    print("Stage 6: Finalizing Video")
    final_video_path = os.path.join(output_dir, "final_output.mp4")
    build_final_video_from_dir(music_dir, video_path, final_video_path)
    print(f"Finished! Video saved to {final_video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args.video, args.outdir)
