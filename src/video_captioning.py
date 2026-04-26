import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

CAPTION_PROMPT = """
You are given a video clip. Carefully watch the entire video.
Your task is to provide a dense, expressive caption that will later be used
to design background music for this scene.

Guidelines:
- Describe what is shown: characters, setting, objects, environment.
- Describe the main actions and motion over time.
- Describe the emotional atmosphere and tension level.
- Highlight cues relevant to music scoring (energy, intensity, pacing, emotional shift).
- Avoid musical terms (no instruments, genre, tempo, BPM, key).
- Write 8–12 concise sentences.

The caption should help a film composer understand how the music should feel.
"""

def get_captioning_model(model_path="OpenGVLab/VideoChat-R1_5"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def inference(video_path, prompt, model, processor, max_new_tokens=1024, device="cuda"):
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                 "video": video_path,
                 "total_pixels": 128 * 12 * 28 * 28, 
                 "min_pixels": 128 * 28 * 28},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def caption_clusters(clusters_path, model, processor, save_path=None):
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    results = []
    for cluster in tqdm(clusters):
        video_path = cluster["path"]
        caption_output = inference(video_path, CAPTION_PROMPT, model, processor)
        results.append({
            "path": video_path,
            "start": cluster["start"],
            "end": cluster["end"],
            "duration": cluster["duration"],
            "caption": caption_output.strip()
        })
    if save_path is None:
        save_path = clusters_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
