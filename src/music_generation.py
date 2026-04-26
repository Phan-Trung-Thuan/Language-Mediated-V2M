import os
import math
import soundfile as sf
import numpy as np
import torch
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from audiocraft.models import MusicGen

def normalize_audio(path_in, path_out=None, target_db=-14):
    y, sr = sf.read(path_in, always_2d=True)
    rms = np.sqrt(np.mean(y**2))
    gain = target_db - (20 * np.log10(rms + 1e-9))
    factor = 10 ** (gain / 20)
    y = y * factor
    if path_out is None:
        path_out = path_in
    sf.write(path_out, y, sr)

def generate_music(model, prompt, duration_s, save_path, prev_audio_path=None, device="cuda"):
    sr = model.sample_rate
    overlap_sec = 5
    overlap_samples = overlap_sec * sr
    
    if prev_audio_path is not None and os.path.exists(prev_audio_path):
        prev_audio, _ = sf.read(prev_audio_path, always_2d=True)
        prev_audio_t = torch.tensor(prev_audio.T).unsqueeze(0).to(device)
        prev_last = prev_audio_t[:, :, -overlap_samples:]
        need_continuation = True
        first_gen_limit = 25.0
    else:
        prev_last = None
        need_continuation = False
        first_gen_limit = 30.0

    if duration_s <= first_gen_limit:
        if need_continuation:
            out = model.generate_continuation(prev_last, sr, descriptions=[prompt])
            out_np = out[0].detach().cpu().numpy()[:, overlap_samples:]
        else:
            out = model.generate(descriptions=[prompt])
            out_np = out[0].detach().cpu().numpy()
        out_np = out_np[:, : int(duration_s * sr)]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, out_np.T, sr, subtype="PCM_16")
        return

    if need_continuation:
        out1 = model.generate_continuation(prev_last, sr, descriptions=[prompt])
        out1_np = out1[0].detach().cpu().numpy()[:, overlap_samples:]
    else:
        out1 = model.generate(descriptions=[prompt])
        out1_np = out1[0].detach().cpu().numpy()

    last5 = out1_np[:, -overlap_samples:]
    last5_t = torch.tensor(last5).unsqueeze(0).to(device)

    out2 = model.generate_continuation(last5_t, sr, descriptions=[prompt])
    out2_np = out2[0].detach().cpu().numpy()[:, overlap_samples:]

    full = np.concatenate([out1_np, out2_np], axis=1)
    max_samples = math.ceil(duration_s * sr)
    full = full[:, :max_samples]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write(save_path, full.T, sr, subtype="PCM_16")

def build_final_video_from_dir(music_dir, full_video_path, output_path, fade_t=1.5):
    import re
    audio_files = [f for f in os.listdir(music_dir) if f.startswith("music_cluster") and f.endswith(".wav")]
    def extract_num(f):
        m = re.search(r'_(\d+)\.', f)
        return int(m.group(1)) if m else 0
    audio_files = sorted(audio_files, key=extract_num)
    if not audio_files:
        raise RuntimeError("No music_cluster_*.wav files found in directory.")
    
    audio_clips = [AudioFileClip(os.path.join(music_dir, a)) for a in audio_files]
    final_audio = audio_clips[0]
    total_audio_duration = audio_clips[0].duration
    
    for i in range(1, len(audio_clips)):
        prev_end = final_audio.duration - fade_t
        final_audio = CompositeAudioClip([
            final_audio,
            audio_clips[i].set_start(prev_end)
        ])
        total_audio_duration += (audio_clips[i].duration)
    
    base_video = VideoFileClip(full_video_path).subclip(0, final_audio.duration)
    final_video = base_video.set_audio(final_audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=base_video.fps)
