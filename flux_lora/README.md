Taken from [https://github.com/huggingface/diffusers/blob/main/examples/flux-control/train_control_flux.py](https://github.com/huggingface/diffusers/blob/main/examples/flux-control/train_control_flux.py)


- Ensure `venv` - `python -m venv venv`
- Activate `venv` - `scripts\activate.bat`
- Ensure `uv` installed

- `hf auth login`


```
python -m pip install --upgrade pip setuptools wheel

uv pip install -r requirements.txt

uv pip uninstall -y diffusers
uv pip install git+https://github.com/huggingface/diffusers.git


uv pip install sentencepiece protobuf
uv pip install --upgrade transformers

uv pip uninstall -y torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

run `run.


python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
