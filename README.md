# VLM Pipeline - Image Significance Analysis

A  pipeline that uses Vision-Language Models (VLM) to analyze image significance in personal photo collections. The system evaluates how meaningful photos are to individuals by analyzing family relationships, event context, and image quality.

## 🚀 Features

- **Image Significance Scoring**: Analyzes photos and provides a significance score (1-100) based on multiple criteria
- **Family Face Recognition**: Compares event photos against known family member reference images
- **Event Context Analysis**: Utilizes event metadata (name, location) for better significance assessment
- **Batch Processing**: Processes entire datasets of organized photo collections
- **Multiple VLM Support**: Works with Hugging Face models and Hyperbolic API

## 📁 Project Structure

```
vlm-pipeline/
├── hf-inference.py            # Basic VLM demo with Hugging Face
├── hyperbolic.py              # Simple Hyperbolic API demo
├── img_importance_analyser.py # Experimental significance analysis pipeline
├── run_pipeline.py            # Main significance analysis pipeline
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── dataset/                   # Data folder
    └── [Person Name]/         # Individual datasets
        ├── [event_name]/      # Event folders
        │   ├── info.json      # Event metadata
        │   └── *.png/jpg      # Event images
        └── [Person]_FamilyFaces/  # Reference family photos
            ├── father.png
            ├── mother.png
            ├── brother.png
            └── ...
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.13 or higher
- API keys for either:
  - Hugging Face (HF_TOKEN)
  - Hyperbolic API (HYPERBOLIC_API_KEY)

### Installation

1. **Clone the repository**

   ```powershell
   git clone <repository-url>
   cd vlm-pipeline
   ```

2. **Create a virtual environment**

   ```powershell
   uv venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```powershell
   uv sync
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   # For Hugging Face inference (not required for img_importace_analyser.py)
   HF_TOKEN=your_huggingface_token_here

   # For Hyperbolic API
   HYPERBOLIC_API_KEY=your_hyperbolic_api_key_here
   ```

### Getting API Keys

#### Hugging Face Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Copy the token to your `.env` file

#### Hyperbolic API Key

1. Sign up at [Hyperbolic](https://www.hyperbolic.ai/)
2. Navigate to API settings
3. Generate a new API key
4. Copy the key to your `.env` file

## 📊 Dataset Structure

The pipeline expects a specific dataset structure:

### Event Folders

Each event should have:

- **Images**: `.png`, `.jpg`, or `.jpeg` files
- **Metadata**: `info.json` file with event information in every event folder

Example `info.json`:

```json
{
  "event": "xxx",
  "location": ["Home", "Resort"]
}
```

### Family Faces Folder

Create a folder named `[Person]_FamilyFaces` containing reference photos:

- `father.png`
- `mother.png`
- `brother.png`
- `aunt.png`
- `uncle_1.png`
- etc.

**Note**: Filenames should indicate the relationship (the system uses filenames to identify family members).

## 🚀 Usage

### Basic VLM Demo

Test the basic VLM functionality:

run `hf-inference.py` or `hyperbolic.py`

This will analyze a sample image using the Qwen2.5-VL model through the chosen inference provider.

**Note**: Update the `img_path` variable in `hyperbolic.py` to point to your image.

### Full Image Significance Analysis

Run the complete significance analysis pipeline:

run `run_pipeline.py`

```powershell
python .\img_importance_analyser_old.py --dataset "<dataset root with event folders and FamilyFaces folder>" --output "<output directory>"
```

The output will be a folder following similar folder sturcture to the dataset, uses the same image name as the file name of a txt with the final score for that picture stored.

## ⚙️ Configuration

### Model Selection

The pipeline supports different VLM models. In `run_pipeline.py`, you can modify:

```python
model = "Qwen/Qwen2.5-VL-7B-Instruct"  # Default model
```

### Scoring Parameters

The significance scoring considers:

- **People Presence (40 points)**: Family members, facial expressions, eye contact
- **Technical Quality (20 points)**: Focus, lighting, composition
- **Event Significance (20 points)**: Special occasions, holidays, milestones
- **Emotional/Scenic Value (20 points)**: Emotions, location, uniqueness

### API Parameters

Adjust API call parameters in the `build_payload()` function:

```python
"max_tokens": 512,
"temperature": 0.1,
"top_p": 0.001,
```

## 📈 Output

The pipeline generates:

1. **Console Output**: Progress updates and significance scores
2. **Legacy Results Files**: Text files with significance scores saved to:
   ```
   dataset/results/[Person Name]/[event_name]/[image_name].txt
   ```

Each result file contains a single number (1-100) representing the image's significance score.

To convert the legacy dataset plus these score files into the PRISM-style `Data/` + `Annotations/` layout, run:

```powershell
python .\scripts\dataset_formatting.py `
  --source-dataset "<original dataset root>" `
  --source-results "<results root>" `
  --output-root "<converted dataset root>"
```

The converter:

- Flattens event folders into split-specific creator folders under `Data/train|val|test/<creator>/`
- Writes JSON sidecars under `Annotations/train|val|test/<creator>/`
- Copies event metadata from `info.json`
- Extracts and normalizes `vlm_score` from legacy `.txt` outputs
- Adds PRISM-friendly fields such as `metadata_text`, `reasoning`, `user_id`, `event_name`, and `location_details`

## API Limitations

- **Hyperbolic API**: Limited concurrent requests and image count per request. Can send at max. 4 images per api call.
- **Hugging Face**: Rate limits may apply based on your subscription tier

## Final Dataset File Structure

```
DatasetRoot/
│
├── Annotations/
│   ├── train/
│   │   ├── Amit Bhadana/
│   │   ├── Ashish Chanchlani/
│   │   ├── Brent Rivera/
│   │   └── ...
│   │
│   ├── val/
│   │   ├── Amit Bhadana/
│   │   ├── Ashish Chanchlani/
│   │   └── ...
│   │
│   └── test/
│       ├── Amit Bhadana/
│       ├── Ashish Chanchlani/
│       └── ...
│
└── Data/
    ├── train/
    │   ├── Amit Bhadana/
    │   ├── Ashish Chanchlani/
    │   └── ...
    │
    ├── val/
    │   ├── Amit Bhadana/
    │   └── ...
    │
    └── test/
        ├── Amit Bhadana/
        └── ...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
