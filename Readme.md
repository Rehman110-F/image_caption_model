# Image Captioning - CNN + Transformer

> ResNet-50 encoder + Transformer decoder trained on MS COCO dataset.
> Converted from a Jupyter notebook into a clean, production-ready project structure.

---

## Project Structure

```
image_captioning/
|
|-- configs/
|   |-- config.yaml               <- all paths and hyperparameters
|   |-- config_loader.py          <- YAML loader
|
|-- src/
|   |-- data/
|   |   |-- prepare.py            <- load COCO pairs, compute mean/std, build vocab
|   |   |-- dataset.py            <- CaptionDataset, collate_function, build_loaders
|   |-- models/
|   |   |-- encoder.py            <- CNNEncoder (ResNet-50)
|   |   |-- decoder.py            <- PositionalEncoding + TransformerDecoder
|   |   |-- captioning_model.py   <- ImageCaptioningModel (encoder + decoder)
|   |-- training/
|   |   |-- train_utils.py        <- generate_square_subsequent_mask, train_one_epoch, validate
|   |-- inference/
|   |   |-- predictor.py          <- generate_caption + CaptionPredictor wrapper
|   |-- api/
|       |-- app.py                <- FastAPI server (POST /caption)
|
|-- scripts/
|   |-- prepare_data.py           <- run once to cache vocab + stats
|   |-- train.py                  <- full training run
|   |-- predict.py                <- CLI inference
|   |-- run_api.py                <- start the API server
|
|-- frontend/
|   |-- index.html                <- drag-and-drop web UI
|
|-- Dockerfile                    <- Docker build file (CPU version)
|-- docker-compose.yml            <- Docker Compose config
|-- .dockerignore                 <- files excluded from Docker image
|-- requirements.txt              <- GPU dependencies
|-- requirements-cpu.txt          <- CPU dependencies (for Docker / no GPU)
|-- data/                         <- put your COCO images + annotations here
|-- saved_models/                 <- checkpoints saved here
|-- README.md
```

---

## Which requirements file should I use?

| Situation | File to use |
|-----------|-------------|
| You have an NVIDIA GPU (CUDA) | `requirements.txt` |
| You have no GPU (CPU only) | `requirements-cpu.txt` |
| Building Docker image | uses `requirements-cpu.txt` automatically |

---

## Setup - Create Environment First

Before installing any packages, create an isolated environment.
This keeps your project dependencies separate from other projects.

### Option A - Conda (Recommended)

```bash
# Create a new conda environment with Python 3.10
conda create -n imgcap python=3.10

# Activate it
conda activate imgcap

# You will see (imgcap) in your terminal now
```

### Option B - Virtual Environment (venv)

```bash
# Create virtual environment
python -m venv imgcap

# Activate it on Windows
imgcap\Scripts\activate

# Activate it on Mac/Linux
source imgcap/bin/activate

# You will see (imgcap) in your terminal now
```

> NOTE: Always make sure your environment is activated before running any commands in this project.
> You will see `(imgcap)` at the start of your terminal line when it is active.

---

## Setup - GPU (NVIDIA CUDA)

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
```
torch==2.6.0
torchvision==0.21.0
Pillow==12.0.0
numpy==2.2.6
tqdm==4.67.3
PyYAML==6.0.3
fastapi==0.135.1
uvicorn==0.41.0
python-multipart==0.0.22
pydantic==2.12.5
matplotlib==3.10.8
starlette==0.52.1
anyio==4.12.1
```

Make sure you have the correct CUDA version of PyTorch installed.
Check your version at: https://pytorch.org/get-started/locally/

```bash
# This project uses CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## Setup - CPU only (No GPU)

```bash
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` contains:
```
torch==2.2.0
torchvision==0.17.0
Pillow==12.0.0
numpy==1.26.0
tqdm==4.67.3
PyYAML==6.0.3
fastapi==0.135.1
uvicorn==0.41.0
python-multipart==0.0.22
pydantic==2.12.5
matplotlib==3.10.8
starlette==0.52.1
anyio==4.12.1
```

> NOTE: Training on CPU will be very slow. GPU is strongly recommended for training.
> For inference (predicting captions on single images) CPU is fine, just slower (5-15 sec per image).

---

## Data Layout

```
data/
|-- images/
|   |-- val2017/               <- *.jpg images from COCO val2017
|-- annotations/
    |-- captions_val2017.json
```

| File | Download Link |
|------|---------------|
| Images (val2017) | http://images.cocodataset.org/zips/val2017.zip |
| Captions | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |

After downloading, update the paths in `configs/config.yaml` to point to your local data.

---

## Usage - Without Docker

### Step 1 - Download trained model weights

Download the trained model weights from Google Drive:

**https://drive.google.com/file/d/1ySPlVzzgmCZdO4S2xRXUmvV-meOVcCYk/view?usp=sharing**

Place the downloaded file into the `saved_models/` folder:
```
saved_models/
    image_caption_model_final.pth   <- put it here
```

---

### Step 2 - Prepare Data (run once)

```bash
python scripts/prepare_data.py
```

Saves `data/vocab.json` and `data/dataset_stats.json`.

---

### Step 2 - Train

> NOTE: Requires GPU. Make sure you installed `requirements.txt` (GPU version).

```bash
python scripts/train.py
```

Resume training from a saved checkpoint:

```bash
python scripts/train.py --resume saved_models/checkpoint_epoch_005.pth
```

---

### Step 3 - Predict (CLI)

```bash
# Provide the full path to your image inside quotes
python scripts/predict.py --image "C:/Users/YourName/Pictures/your_image.jpg"
```

---

### Step 4 - API + Frontend

Start the API server:

```bash
python scripts/run_api.py
```

Then open `frontend/index.html` by double-clicking it in your file explorer.

> Make sure the API URL box in the frontend shows http://127.0.0.1:8000 before uploading an image.

Swagger API docs: http://127.0.0.1:8000/docs

---

## Usage - With Docker

> Docker uses `requirements-cpu.txt` automatically. No GPU needed on the Docker machine.

### How Docker works for this project

```
Your Machine
|-- saved_models/    <- your trained model  (mounted into container)
|-- data/            <- vocab + stats files (mounted into container)
|-- Docker Container
        |-- Python 3.10
        |-- PyTorch 2.1.0 (CPU)
        |-- All packages from requirements-cpu.txt
        |-- runs: python scripts/run_api.py
                        |
                        v
               API at http://localhost:8000
```

---

### Step 1 - Install Docker

Download Docker Desktop from: https://www.docker.com/products/docker-desktop/

Verify it works:
```bash
docker --version
docker ps
```

---

### Step 2 - Download model weights

Download the trained model weights from Google Drive:

**https://drive.google.com/file/d/1ySPlVzzgmCZdO4S2xRXUmvV-meOVcCYk/view?usp=sharing**

After downloading, place the file here:
```
saved_models/
    image_caption_model_final.pth   <- put the downloaded file here
```

Also make sure these files exist (built by prepare_data.py):
```
data/vocab.json
data/dataset_stats.json
```

---

### Step 3 - Build Docker image

```bash
docker build -t image-captioning .
```

This takes 10-15 minutes the first time (downloads Python and installs packages).

---

### Step 4 - Run with Docker Compose

```bash
docker-compose up
```

To stop:
```bash
docker-compose down
```

---

### Step 5 - Open the frontend

Double-click `frontend/index.html` in your file explorer.
Make sure the API URL shows `http://127.0.0.1:8000` then upload an image.

---

### Docker files explained

| File | Purpose |
|------|---------|
| `Dockerfile` | builds the image using python:3.10-slim and requirements-cpu.txt |
| `docker-compose.yml` | mounts data/ and saved_models/ and runs the container |
| `.dockerignore` | excludes large files from being copied into the image |
| `requirements-cpu.txt` | CPU-only packages used inside Docker |

---

## API

```bash
curl -X POST http://127.0.0.1:8000/caption \
     -F "file=@photo.jpg"
```

Response:

```json
{
  "caption": "a group of people standing in a field"
}
```

---

## Tips

- Always run all scripts from the project root directory, not from inside the scripts/ folder.
- The pretrained deprecation warning from torchvision is harmless - the model loads correctly.
- vocab.json and dataset_stats.json are cached after first prepare_data.py run - no need to rerun.
- A checkpoint is saved after every epoch inside saved_models/ so you can resume training anytime.
- For CPU-only inference the model automatically falls back to CPU - no code changes needed.
- Docker uses the CPU version of PyTorch - do not use Docker for training, only for inference and API.

---

## Author

**Muhammad Rehman Ashraf**

---

*Built with PyTorch - torchvision - FastAPI - MS COCO*