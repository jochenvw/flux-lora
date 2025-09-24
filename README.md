# FLUX LoRA Training Pipeline

This repository provides a complete pipeline for training Low Rank Adaptation (LoRA) models on FLUX using food photography images. The workflow automates image processing, AI-powered labeling, caption generation, and model training for creating custom diffusion models.

## Overview

This pipeline is specifically designed for food photography datasets and includes:

1. **Image Conversion**: Convert various image formats (AVIF, WebP, etc.) to JPG
2. **AI-Powered Labeling**: Use Azure OpenAI/GPT-4o-mini to analyze and label images with visual attributes
3. **Caption Generation**: Generate training captions from structured labels
4. **LoRA Training**: Train custom LoRA adapters for FLUX models using the processed dataset

## Features

- **Automated Image Processing**: Batch convert images to standard formats
- **Intelligent Image Analysis**: AI-powered labeling of visual attributes including:
  - Perspective (top-down, eye-level, low-angle)
  - Background (setting, simplicity vs clutter)
  - Lighting (soft, harsh, high-key, low-key, golden-hour)
  - Color palette (muted, vibrant, pastel, monochrome)
  - Texture treatment (painterly, grainy, smooth)
  - Composition (centered subject, rule of thirds, negative space)
  - Mood (nostalgic, cheerful, moody, futuristic)
- **Caption Generation**: Transform structured labels into training-ready captions
- **Resume Capability**: Skip already processed images to resume interrupted workflows
- **Progress Tracking**: Rich console output with progress bars and logging

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Azure OpenAI API access or OpenAI API key
- Required Python packages (installed automatically in devcontainer)

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jochenvw/flux-lora.git
   cd flux-lora
   ```

2. **Environment Setup**:
   The repository includes a devcontainer configuration for easy setup. Alternatively, install dependencies manually:
   ```bash
   pip install pillow pillow-avif-plugin openai diffusers==0.30.2 huggingface_hub==0.24.6 accelerate peft datasets python-dotenv rich tqdm ipywidgets
   ```

3. **Configure API Access**:
   Copy `.env.template` to `.env` and fill in your API credentials:
   ```bash
   cp .env.template .env
   ```
   
   Edit `.env` with your Azure OpenAI or OpenAI credentials:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   ```
   
   Or for regular OpenAI:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

The entire pipeline is contained in the `process_images.ipynb` Jupyter notebook. Run the cells sequentially:

### Step 1: Image Conversion
Place your source images in the `images-new/` directory. The notebook will:
- Convert all images to JPG format
- Resize to appropriate dimensions
- Save converted images to `work/converted/`

### Step 2: AI-Powered Image Labeling
The system analyzes each image using GPT-4o-mini to identify:
- Food items and main subjects
- Visual styling attributes (lighting, composition, mood, etc.)
- Results are saved as JSONL format in `work/labels/`

### Step 3: Caption Generation
Transforms structured labels into natural language captions:
- Combines food items with visual attributes
- Generates training-ready text descriptions
- Saves caption files alongside images in `work/dataset/new/`

### Step 4: LoRA Training
Configures and trains custom LoRA adapters:
- Uses the processed image-caption pairs
- Supports various training configurations
- Integrates with Hugging Face diffusers library

## Output Structure

```
work/
├── converted/          # JPG converted images
├── labels/            # JSONL files with AI-generated labels
└── dataset/
    └── new/           # Training dataset (images + caption .txt files)
```

## Example Output

**Generated Labels** (JSONL format):
```json
{"filename": "zoete-aardappel-met-kip-en-gebakken-paksoi.jpg", "labels": {"perspective": "top-down", "lighting": "soft", "composition": "rule of thirds", "mood": "cheerful", "background": "simple setting"}}
```

**Generated Caption** (TXT file):
```
composition: rule of thirds, perspective: top-down, lighting: soft, background: simple setting, mood: cheerful.
```

## Configuration

Key configurable parameters in the notebook:

- `MAX_IMAGES_TO_LABEL`: Limit number of images to process (default: 10)
- `OPENAI_MODEL`: AI model for labeling (default: "gpt-4o-mini")
- `LABEL_SCHEMA`: Visual attributes to extract from images
- Training parameters: learning rate, batch size, mixed precision, etc.

## Model Training Settings

The pipeline supports various training configurations:
- **Base Model**: Stable Diffusion v1.5 (configurable)
- **Mixed Precision**: FP16 support for memory efficiency
- **Optimizer**: 8-bit Adam optimizer option
- **Batch Size**: Configurable training batch size
- **Learning Rate**: Adjustable for different datasets

## Resuming Work

The pipeline automatically detects and skips already processed images, allowing you to:
- Resume interrupted labeling sessions
- Add new images to existing datasets
- Reprocess specific components without starting over

## Troubleshooting

- **PyTorch Version Issues**: The notebook includes automatic version patching for development builds
- **Memory Issues**: Reduce batch size or enable mixed precision training
- **API Rate Limits**: The system includes retry logic and progress tracking
- **Image Format Support**: Ensure `pillow-avif-plugin` is installed for AVIF support

## Contributing

This pipeline is designed for food photography but can be adapted for other domains by:
- Modifying the `LABEL_SCHEMA` for your specific attributes
- Adjusting the caption generation templates
- Customizing the training parameters

## License

[Add your license information here]