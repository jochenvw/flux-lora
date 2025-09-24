# FLUX vs Stable Diffusion LoRA Training Comparison

This document compares the two notebooks in this repository:
- `process_images.ipynb` - Original Stable Diffusion 1.5 LoRA training
- `process_images_flux.ipynb` - New FLUX.1-dev LoRA training

## Key Architectural Differences

| Component | Stable Diffusion 1.5 | FLUX.1-dev |
|-----------|----------------------|------------|
| **Base Model** | `runwayml/stable-diffusion-v1-5` | `black-forest-labs/FLUX.1-dev` |
| **Text Encoder** | CLIPTextModel | T5EncoderModel |
| **Main Architecture** | UNet2DConditionModel | FluxTransformer2DModel |
| **Scheduler** | DDPMScheduler (denoising) | FlowMatchEulerDiscreteScheduler (flow matching) |
| **Pipeline** | StableDiffusionImg2ImgPipeline | FluxPipeline (text2img only) |

## Training Parameter Differences

| Parameter | Stable Diffusion 1.5 | FLUX.1-dev | Reasoning |
|-----------|----------------------|------------|-----------|
| **Resolution** | 512x512 | 1024x1024 | FLUX trained for higher res |
| **Batch Size** | 2 | 1 | FLUX requires more VRAM |
| **Mixed Precision** | fp16 | bf16 | Better numerical stability for FLUX |
| **LoRA Rank** | 8 | 16 | Higher rank for better quality |
| **LoRA Alpha** | 16 | 32 | Scaled with rank |
| **LoRA Dropout** | 0.05 | 0.1 | Higher dropout for regularization |

## Generation Parameter Differences

| Parameter | Stable Diffusion 1.5 | FLUX.1-dev | Reasoning |
|-----------|----------------------|------------|-----------|
| **Guidance Scale** | 7.0 | 3.5 | FLUX works better with lower guidance |
| **Inference Steps** | 30 | 28 | FLUX optimal step count |
| **Output Size** | 512x512 | 1024x1024 | Native FLUX resolution |
| **Generation Type** | img2img with strength | text2img only | FLUX pipeline difference |

## Performance Considerations

### FLUX Advantages:
- ✅ Higher resolution output (1024x1024 vs 512x512)
- ✅ Better text understanding (T5 vs CLIP)
- ✅ More modern architecture (Transformer vs UNet)
- ✅ Flow matching training (vs denoising diffusion)

### FLUX Requirements:
- ⚠️ More VRAM needed (8GB+ recommended)
- ⚠️ Requires HuggingFace access to FLUX.1-dev
- ⚠️ Longer training time due to higher resolution
- ⚠️ More compute-intensive generation

## Usage Recommendations

### Use Stable Diffusion 1.5 (`process_images.ipynb`) when:
- Limited VRAM (4-6GB)
- Quick prototyping needed
- img2img workflow required
- Working with existing SD1.5 ecosystem

### Use FLUX.1-dev (`process_images_flux.ipynb`) when:
- RTX4090 or similar high-end GPU available
- Best quality results needed
- Working with complex text prompts
- Taking advantage of latest architecture

## Code Structure Similarities

Both notebooks maintain the same overall workflow:
1. **Image Conversion** - Convert various formats to JPG
2. **AI-Powered Labeling** - Use OpenAI to analyze images
3. **Caption Generation** - Create training captions
4. **LoRA Training** - Train custom adapters
5. **Image Generation** - Generate images with trained LoRA

The main differences are in the model loading, training loop, and generation sections, while the data preprocessing remains largely the same.