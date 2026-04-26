# Language-Mediated Video-to-Music Generation

This repository contains the official implementation of the graduation thesis: **"Developing a web application for Music Generation from video using Natural Language as Intermediary by open source AI Models"** by Phan Trung Thuan (B2111957), Class 2021-2025 (K47), Information Technology (High-Quality Program), College of Information and Communication Technology, Can Tho University.

Advisor: **Dr. Lam Nhut Khang**

## Abstract

This project presents a language-mediated framework for video-to-music generation, aiming to automatically generate background music that is semantically aligned with video content. Unlike conventional approaches that directly fuse visual and audio features, this system employs natural language as an intermediate representation to bridge video understanding and music generation. The framework integrates scene segmentation, video captioning, music feature inference, and prompt-based music generation using open-source models, enabling improved interpretability and controllability.

## Pipeline Overview

The system decomposes the task into interpretable stages:

1.  **Scene Detection & Segmentation**: Uses `PySceneDetect` to identify visual discontinuities and `CLIP` embeddings to cluster semantically similar consecutive scenes.
2.  **Video Captioning**: Employs `VideoChat-R1.5` (an enhanced variant of Qwen2.5-VL) to generate dense, expressive captions (5-8 sentences) for each video segment.
3.  **Music Feature Proposal**: A Small Language Model (`Qwen3-4B-Instruct`) translates textual video summaries into structured music-theoretical features (Genre, Mood, Style, Instruments, Context).
4.  **Prompt Preparation**: The same SLM refines these features into a concise, `MusicGen`-compatible natural language prompt.
5.  **Music Generation**: Relies on Meta's `MusicGen` with a continuation strategy to maintain temporal coherence across segments (using a 10s overlap window).
6.  **Output Rendering**: Integrates generated audio segments with the original video using `MoviePy`, applying crossfades for smooth auditory transitions.

## Usage

To run the full video-to-music generation pipeline, use the `main.py` script:

```bash
python src/main.py --video "path/to/your/video.mp4" --outdir "output_directory"
```

Ensure all required models and dependencies (PySceneDetect, Transformers, Audiocraft, MoviePy) are installed.

## Tech Stack

| Component            | Model / Tool                    | Parameters (B) |
| :------------------- | :------------------------------ | :------------- |
| **Clustering**       | `openai/clip-vit-large-patch14` | 0.4            |
| **Captioning**       | `OpenGVLab/VideoChat-R1_5-7B`   | 7.0            |
| **Music Reasoning**  | `Qwen/Qwen3-4B-Instruct-2507`   | 4.0            |
| **Music Generation** | `facebook/musicgen-medium`      | 1.5            |

## Experimental Results

### 1. Music Quality Metrics

Evaluates the intrinsic acoustic properties of the generated music.
| Models | THD ↓ | SF ↓ | SCV ↑ | ZCR ↓ |
| :--- | :--- | :--- | :--- | :--- |
| GroundTruth | 0.1603 | 0.0167 | 718.7996 | 0.0648 |
| GVMGen [9] | **0.1339** | **0.0135** | **709.3384** | **0.0639** |
| **Proposed pipeline** | 0.2390 | 0.0144 | 564.8818 | 0.0693 |

### 2. Video-Music Relationship Metrics

Quantifies semantic, rhythmic, and temporal alignment between video and music.
| Models | CMEC ↑ | BMS ↑ | CMSA ↑ | SSM ↑ |
| :--- | :--- | :--- | :--- | :--- |
| GroundTruth | 0.2979 | 0.2436 | 0.1696 | 0.2740 |
| GVMGen | 0.1469 | 0.0913 | 0.1430 | 0.1188 |
| **Proposed pipeline** | **0.2456** | 0.0271 | **0.1816** | **0.1313** |
_CMEC: Cross-Modal Embedding Correlation, BMS: Beat-Motion Synchrony, CMSA: Cross-Modal Similarity Alignment, SSM: Self-Similarity Matrix._

### 3. Computational Cost Comparison

The proposed system is significantly more efficient than the GVMGen baseline.
| Average values of test set | GVMGen | **Proposed pipeline** |
| :--- | :--- | :--- |
| Video duration | 198.54s | 198.54s |
| Computational time (seconds) | 8197.75s | **2245.19s** (~3.6x faster) |
| Memory usage (GB VRAM) | 20.93 | **10.57** (~50% reduction) |
| Total Parameters (Billions) | **2.2** | 12.9 |

## Repository Structure

- `src/`: Python source code for the modular pipeline.
- `docs/`: Thesis PDF, presentation, and time estimation spreadsheet.
- `data/`: Manifests and cluster information.
- `inference_videos/`: Example outputs comparing different models.
- `notebooks/`: Original Jupyter notebook for the full pipeline.

## Conclusion

The language-mediated approach effectively preserves high-level semantics from video content. While beat-level synchronization (BMS) is a trade-off, the system achieves superior semantic alignment (CMEC/CMSA) and structural consistency (SSM) compared to state-of-the-art baselines like GVMGen, while being much more computationally efficient for web deployment.
