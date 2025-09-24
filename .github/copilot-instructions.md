# FLUX LoRA Training Pipeline

FLUX LoRA is a machine learning project for training Low Rank Adaptation (LoRA) models for the FLUX image generation model. The main workflow processes food images, generates captions using AI, and trains a LoRA adapter for style-specific image generation.

**ALWAYS reference these instructions first and fallback to additional search and context gathering only if the information in the instructions is incomplete or found to be in error.**

## Working Effectively

### Environment Setup
- **CRITICAL**: This project requires GPU support for optimal performance. Use the provided devcontainer with NVIDIA PyTorch base image.
- **CRITICAL**: Set up environment configuration:
  - `cp .env.template .env`
  - Edit `.env` file with your API credentials (Azure OpenAI or OpenAI API keys required for image labeling)
- **NEVER CANCEL**: Package installation takes 60-90 seconds. Wait for completion:
  - `pip install pillow pillow-avif-plugin openai "diffusers==0.30.2" "huggingface_hub==0.24.6" accelerate peft datasets python-dotenv rich tqdm ipywidgets jupyter`
  - Set timeout to 120+ seconds for this command
- **NEVER CANCEL**: Library imports take 4-5 seconds due to ML dependencies. This is normal.

### Core Workflow
- **Primary Interface**: `process_images.ipynb` Jupyter notebook (14 cells, 8 code cells)
- **NEVER CANCEL**: Notebook execution can take 30+ minutes depending on:
  - Image processing steps
  - AI model loading (requires internet access to HuggingFace Hub)
  - Caption generation via OpenAI/Azure OpenAI APIs
  - LoRA training (GPU-intensive, can take hours)
- **NEVER CANCEL**: Model downloads from HuggingFace can take 10-30 minutes for large FLUX models. Set timeouts to 45+ minutes.

### Directory Structure
```
├── .devcontainer/          # Dev container configuration (NVIDIA PyTorch)
├── .env.template          # Environment configuration template
├── process_images.ipynb   # Main workflow notebook
├── images-new/           # Input images (AVIF, WebP, JPG formats)
├── images-old/           # Additional input images
└── work/
    ├── dataset/new/      # Processed image-caption pairs (47+ files)
    ├── labels/          # Generated captions and metadata
    └── converted/       # Format-converted images
```

### Running the Workflow
1. **Validate environment setup first**:
   - Run basic import test to verify all dependencies work (takes 4-5 seconds)
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Verify dataset structure: ensure `work/dataset/new/` contains paired .jpg/.txt files

2. **Configure API Access**:
   - **REQUIRED**: Set up `.env` file with either Azure OpenAI or OpenAI API credentials
   - Without API access, image captioning will fail
   - Test API connectivity before running full workflow

3. **Execute notebook**:
   - Use Jupyter: `jupyter notebook process_images.ipynb`
   - Or use VS Code with Jupyter extension in devcontainer
   - **Execute cells sequentially** - later cells depend on earlier setup

4. **Monitor progress**:
   - Watch for "rich" formatted progress bars during image processing
   - Model loading shows progress via transformers library
   - Caption generation shows per-image progress

## Validation and Testing

### Always validate your changes with these scenarios:
1. **Environment Test**: 
   - Import all key libraries (torch, diffusers, accelerate, peft, transformers, huggingface_hub, openai, rich, datasets, PIL)
   - Verify takes 4-5 seconds, no errors
   - Check CUDA availability status

2. **Data Pipeline Test**:
   - Verify `work/dataset/new/` contains paired image and text files
   - Sample text file format: "composition: centered subject, perspective: eye-level, lighting: soft, background: simple setting, mood: cheerful."
   - Test image loading from various formats (AVIF, WebP, JPG)

3. **API Connectivity Test**:
   - Test OpenAI/Azure OpenAI connection with small request
   - Verify environment variables are loaded correctly

4. **Notebook Execution Test**:
   - Run first few cells to validate setup
   - **Do NOT run full training without intent** - it's resource intensive

### Common Issues and Solutions
- **Import Errors**: Reinstall packages, check for network timeouts during pip install
- **AVIF Support Issues**: pillow-avif-plugin may fail to install due to network issues. AVIF support is helpful but not strictly required.
- **CUDA Not Available**: Expected in non-GPU environments. Workflow will use CPU (much slower).
- **HuggingFace Hub Errors**: Requires internet access. Models cached locally after first download.
- **API Rate Limits**: OpenAI/Azure OpenAI have rate limits. Implement delays between requests.

## Performance and Timing Expectations

- **Package Installation**: 60-90 seconds - **NEVER CANCEL**
- **Library Imports**: 4-5 seconds - **NEVER CANCEL** 
- **Model Downloads**: 10-30 minutes for large models - **NEVER CANCEL**, set 45+ minute timeouts
- **Image Processing**: 1-2 minutes per image for captioning
- **LoRA Training**: 30+ minutes to several hours depending on GPU and dataset size - **NEVER CANCEL**

**Critical Timing Rules**:
- Set explicit timeouts of 120+ seconds for package installation
- Set explicit timeouts of 60+ minutes for model loading and training operations
- Always wait for completion of long-running ML operations
- Monitor memory usage during training - may require 8-16GB+ GPU memory

## Development Tips

- **Use DevContainer**: Provides consistent NVIDIA PyTorch environment with GPU support
- **Incremental Testing**: Test each notebook cell individually before full execution
- **Resource Monitoring**: Watch GPU memory usage during model operations
- **Checkpoint Frequently**: Save intermediate results to avoid losing progress
- **API Cost Management**: Monitor OpenAI/Azure costs during caption generation phases

## Common Commands Reference

```bash
# Environment validation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Package installation (NEVER CANCEL - takes 60-90 seconds)
pip install pillow pillow-avif-plugin openai "diffusers==0.30.2" "huggingface_hub==0.24.6" accelerate peft datasets python-dotenv rich tqdm ipywidgets jupyter

# Start Jupyter notebook
jupyter notebook process_images.ipynb

# Check dataset structure
ls -la work/dataset/new/ | wc -l  # Should show 47+ paired files

# Test import speed (should take 4-5 seconds)
time python -c "from diffusers import DiffusionPipeline; from peft import LoraConfig; import torch; print('Imports successful')"
```

## Quick Start Validation

After cloning the repository, validate your setup works:

```bash
# 1. Set up environment (NEVER CANCEL - takes 60-90 seconds)
cp .env.template .env
pip install pillow pillow-avif-plugin openai "diffusers==0.30.2" "huggingface_hub==0.24.6" accelerate peft datasets python-dotenv rich tqdm ipywidgets jupyter

# 2. Test imports (should take 4-5 seconds)
python -c "import torch; from diffusers import DiffusionPipeline; from peft import LoraConfig; print('✅ Ready to go!')"

# 3. Check data structure (should show 47+ paired files)
ls -la work/dataset/new/ | wc -l

# 4. Start notebook
jupyter notebook process_images.ipynb
```

If all commands complete successfully, you're ready to use the workflow.

## File Structure Summary

Key files you'll work with frequently:
- `process_images.ipynb` - Main workflow (always start here)
- `.env` - API configuration (copy from .env.template)
- `work/dataset/new/` - Training data pairs (images + captions)
- `.devcontainer/` - Container setup with GPU support