import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_qwen_model(model_name="Qwen/Qwen3-4B-Instruct-2507"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model, tokenizer

def qwen3_reason_music(caption: str, model, tokenizer, max_new_tokens=32768):
    prompt = f"""You are a professional film composer and music director.
Analyze the following video scene description and reason carefully
about what kind of background music best supports the scene.

Scene description:
\"\"\"{caption}\"\"\"

After your reasoning, produce the final answer.
Rules for final answer:
- Output format MUST be exactly as specified.
- Feature values must be short (1–8 words).
- Do NOT include tempo, BPM, key, chords, or harmony.
- MusicGen prompt must be 1–2 sentences.
- Include rhythm and energy cues.
- Avoid words implying silence or minimalism (soft, faint, distant, subtle).
- No explanations outside the specified fields.

Final output format:
Genre:
Mood:
Style:
Instruments:
Context:
MusicGenPrompt:
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668) # </think> token
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    final_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, final_content

def parse_final_music_output(text: str):
    data = {"music_features": {"Genre": "", "Mood": "", "Style": "", "Instruments": "", "Context": ""}, "musicgen_prompt": ""}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in data["music_features"]:
            data["music_features"][k] = v
        elif k == "MusicGenPrompt":
            data["musicgen_prompt"] = v
    return data

def suggest_music_features_and_prompt(clusters_path, model, tokenizer, save_path=None):
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    for idx, cluster in enumerate(clusters, 1):
        thinking, final_answer = qwen3_reason_music(cluster["caption"], model, tokenizer)
        parsed = parse_final_music_output(final_answer)
        cluster["music_features"] = parsed["music_features"]
        cluster["musicgen_prompt"] = parsed["musicgen_prompt"]
        cluster["thinking"] = thinking
    if save_path is None:
        save_path = clusters_path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    return clusters
