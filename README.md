# World Architectural Buildings Dataset (FGIC) for Multi‑Class Image Classification

![Dataset Banner](https://huggingface.co/datasets/0xgr3y/arch-building-dataset/resolve/main/greyscope-labs-architecture-buildings-dataset.jpg)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-0xgr3y%2Farch--building--dataset-yellow)](https://huggingface.co/datasets/0xgr3y/arch-building-dataset)
[![Model Training](https://img.shields.io/badge/Model-0xgr3y%2FArch--Building--Image--Classification-blue)](https://huggingface.co/0xgr3y/Arch-Building-Image-Classification)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Space-green)](https://huggingface.co/spaces/0xgr3y/arch-building-classifier)

A balanced, multi-class image classification dataset of world architectural buildings — 13,440 images across 8 classes, sourced from Pexels with dual deduplication (SHA256 + pHash).

## Dataset Description

| Attribute | Value |
|-----------|-------|
| **Curated by** | Saugani |
| **Homepage** | [Greyscope Labs](https://greyscope.xyz) |
| **License** | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (Pexels License — free for commercial use, no attribution required) |
| **Size** | 13,440 images |
| **Classes** | 8 (balanced, 1,680 per class) |
| **Resolution** | Max 720px (longest side), JPEG format |
| **Source** | [Pexels.com](https://www.pexels.com) — authentic photographer-contributed photos |
| **Deduplication** | SHA256 (exact) + pHash (perceptual, threshold ≤ 8) + Human Loop |
| **Contact** | team@greyscope.xyz |

## Classes

| Class | Count | Description |
|-------|-------|-------------|
| 🏚️ **barn** | 1,680 | Traditional wooden barn architecture — residential and storage buildings |
| 🌉 **bridge** | 1,680 | Various bridge architectures (suspension, arch, truss, cable-stayed) |
| 🏰 **castle** | 1,680 | Medieval and modern castle structures, European fortifications |
| 🕌 **mosque** | 1,680 | Islamic mosque architecture (domes, minarets, ornamental facades) |
| 🏙️ **skyscraper** | 1,680 | Tall commercial and residential buildings, city skylines |
| 🏟️ **stadium** | 1,680 | Sports arenas, amphitheaters, large-capacity venues |
| 🛕 **temple** | 1,680 | Religious temple structures (Buddhist, Hindu, Asian) |
| 🎡 **windmill** | 1,680 | Windmill structures (traditional, tower, Dutch) |
| **Total** | **13,440** | All datasets are finalized for curation |

## Data Collection & Processing

### Source

All images sourced from **[Pexels.com](https://www.pexels.com)** — a free stock photo platform where photographers upload authentic original works. Pexels was chosen for:

- **Relevance** — Search results are highly accurate to keywords, producing real building photos without AI illustrations or vector graphics
- **Authentic quality** — All photos uploaded by real Pexels users (no AI-generated content)
- **License** — Pexels License permits free commercial use without attribution
- **Visual consistency** — Single source reduces cross-platform noise

### Collection Pipeline — Multi-Mode Scraping Cascade

Images were collected using `pexels_scraper.py` with a three-mode architecture and automatic fallback:

**Mode 1 — Pexels API Key (primary):**
- REST API `https://api.pexels.com/v1/search` per developer documentation
- Rate limit: 25,000 requests (via API key)
- Pagination: 80 photos/page, max ~6 pages per query (= 480 URLs)
- Two orientations per keyword: *landscape* and *portrait* — doubles unique results
- `X-Ratelimit-Remaining` header monitored to avoid rate limit exceeded

**Mode 2 — BrightData (supplementary):**
- `ChromiumRemoteConnection` from [BrightData](https://brightdata.com) Web Access APIs with proxy
- Scrolls Pexels search pages until no new content (max 80 scrolls)
- Extracts photo IDs from `a[href*='/photo/']` links, constructs medium-resolution image URLs
- Premium account (paid)

**Mode 3 — Selenium (fallback):**
- Headless Chrome local as last resort if Mode 1 and Mode 2 fail

**Workflow per keyword:**
```
Keyword → [Pexels API: landscape + portrait (2 orientations)]
        → [BrightData: scroll search pages if API insufficient]
        → [Selenium: fallback last resort]
        → Collect all URLs → shuffle → parallel download (12 threads)
        → validate → compress → dedup SHA256 + perceptual pHash → save to dataset/
```

### Search Keywords

| Class | Keywords |
|-------|----------|
| 🏚️ **barn** | old wooden barn, barn homes, barn architecture, red barn |
| 🌉 **bridge** | bridge, bridge architecture, bridge landscape, bridge building, bridge at night |
| 🏰 **castle** | castle architecture, medieval castle, castle building, european castle |
| 🕌 **mosque** | mosque architecture, mosque exterior, islamic mosque, beautiful mosque |
| 🏙️ **skyscraper** | skyscraper architecture, skyscraper building, tall building, city skyline, skyscraper at night |
| 🏟️ **stadium** | stadium design, football stadium, arena stadium, stadium at night, stadium architecture |
| 🛕 **temple** | temple architecture, ancient temple, temple building, buddhist temple, temple at night |
| 🎡 **windmill** | windmill architecture, traditional windmill, tower mill, windmill exterior, dutch windmill |

### Compression & Format

| Parameter | Value |
|-----------|-------|
| Max resolution | 720px (longest side), min 200px (shortest side) |
| Max file size | 200 KB per image |
| Format | JPEG |
| Compression | Adaptive quality (start: 85, step: −5, min: 40) until file ≤ 200 KB |
| Resize method | Image.Resampling.LANCZOS (anti-aliased) |

### Quality Filters

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Minimum resolution | 150 × 150 px | Remove thumbnails and icons |
| Aspect ratio | 0.28 – 3.50 | Remove extreme panoramas and strips |
| Color standard deviation | ≥ 10.0 | Remove near-monochrome images |
| Minimum file size | 5 KB | Remove corrupt/stub files |
| Maximum file size | 4,096 KB | Remove uncompressed originals |
| File extension | Skip .svg, .gif, .webm, .mp4, .ico | Ensure JPEG-only dataset |
| URL pattern | Skip logos, avatars, icons, AI art, illustrations, people, badges, buttons, emojis, text overlays | Remove non-photographic content |

### Deduplication

| Method | Threshold | Purpose | Artifact |
|--------|-----------|---------|----------|
| **pHash** (perceptual hash, 16×16) | Distance ≤ 8 | Remove near-duplicate images (visual similarity) | `pexels_scraper.py` — implementation; `pexels_checkpoint.json` — hash records |
| **SHA256** | Exact match | Remove byte-identical duplicates | `pexels_scraper.py` — implementation; `pexels_checkpoint.json` — hash records |
| **Search Keywords** | Per-class | Keyword-driven collection with class-specific queries | `config_images.json` — keyword config; `pexels_scraper.py` — scraper script |
| **Collection Checkpoint** | Per-batch | Resume-capable checkpoint tracking per class — prevents re-downloading across sessions | `pexels_checkpoint.json` — checkpoint data; `pexels_scraper.py` — checkpoint logic |
| **Human Curation** | Full review | Final visual annotation and selection by domain expert | — |

### Resolution Distribution

| Dimension | Range |
|-----------|-------|
| Width | 256 – 889 px |
| Height | 160 – 1140 px |
| Unique resolutions | 773 distinct (train split) |

## Dataset Structure

```
dataset/
├── barn/         barn_00000.jpg – barn_01679.jpg
├── bridge/       bridge_00000.jpg – bridge_01679.jpg
├── castle/       castle_00000.jpg – castle_01679.jpg
├── mosque/       mosque_00000.jpg – mosque_01679.jpg
├── skyscraper/   skyscraper_00000.jpg – skyscraper_01679.jpg
├── stadium/      stadium_00000.jpg – stadium_01679.jpg
├── temple/       temple_00000.jpg – temple_01679.jpg
└── windmill/     windmill_00000.jpg – windmill_01679.jpg
```

- **Naming convention:** `{class}_{5-digit-sequence}.jpg` (sequential, zero-padded)
- **File format:** JPEG (all images)
- **Total files:** 13,440

## Dataset Splits

The recommended split is **80% / 10% / 10%** using [split-folders](https://pypi.org/project/split-folders/) with **seed=42**:

| Split | Images | Per Class |
|-------|--------|-----------|
| Train | 10,752 | 1,344 |
| Validation | 1,344 | 168 |
| Test | 1,344 | 168 |
| **Total** | **13,440** | **1,680** |

## Dataset Comparison

This dataset fills a gap in architectural image classification benchmarks:

| Dataset | Classes | Images | Domain | Limitation |
|---------|---------|--------|--------|------------|
| ImageNet | 1000 | 1.2M | General objects | Few architectural classes, not FGIC-focused |
| Places365 | 365 | 1.8M | Scene recognition | Scene-level, not building-type-level |
| Oxford Buildings | 5K | 5K | Oxford-specific | Single city, limited diversity |
| **This dataset** | **8** | **13,440** | **Global architecture** | **Balanced, multi-cultural, FGIC-focused** |

Unlike ImageNet (which mixes buildings into "castle", "church", etc.) or Places365 (which classifies scenes, not building types), this dataset provides **fine-grained architectural type classification** with balanced classes covering diverse global architecture (European castles, Islamic mosques, Asian temples, Dutch windmills, etc.).

## Class Selection Rationale

The 8 classes were selected based on:

1. **Architectural distinctiveness** — each class has visually discriminative features (domes for mosques, verticality for skyscrapers, blades for windmills)
2. **Cultural diversity** — spans European (barn, castle), Islamic (mosque), Asian (temple), modern (skyscraper, stadium), and utilitarian (bridge, windmill) architecture
3. **FGIC challenge level** — some pairs (temple/mosque, castle/stadium) share visual features, providing inter-class confusion for evaluation
4. **Data availability** — sufficient images on Pexels to achieve balanced 1,680 images per class

## How to Use — Scraping Guide

This repository includes the full scraping toolkit to reproduce or extend the dataset.

### Prerequisites

```bash
pip install -r requirements.txt
```

Dependencies: `requests`, `selenium`, `Pillow`, `imagehash`, `numpy`, `PyWavelets`

### API Configuration

Create a `.env` file (if need a powerful scraping, setup it):

```bash
echo "PEXELS_API_KEY=your_key_here" > .env
echo "BRIGHTDATA_AUTH=your_costumer_zone_browser" >> .env
```

- **PEXELS_API_KEY** — obtain from developer portal [Pexels API-Key](https://www.pexels.com/api/) (free, 25,000 requests)
- **BRIGHTDATA_AUTH** — Good tools premium handle, only needed for Mode 2, [BrightData SDK](https://www.brightdata.com/) 

### Scraping Commands

All configuration is controlled by `config_images.json` — edit this file to change classes, keywords, target counts, compression, or dedup thresholds without modifying Python code.

```bash
# Scrape full dataset (target: 1,680 images per class, total 13,440)
python3 pexels_scraper.py

# View dataset statistics
python3 pexels_scraper.py --stats

# Recompress all images (re-apply compression to existing files)
python3 pexels_scraper.py --recompress

# Preview visual duplicates without deleting (dry run)
python3 pexels_scraper.py --dedup --dry-run

# Remove visual duplicates (pHash, threshold=8)
python3 pexels_scraper.py --dedup
```

### Scraper Features

- ✅ 8 architectural classes with 2–5 search keywords per class (`config_images.json`)
- ✅ Real-time perceptual deduplication during download — each image checked against pHash (threshold=8) within same class
- ✅ Post-process deduplication via `--dedup` CLI flag for cross-class cleanup
- ✅ SHA256 hash deduplication (cross-class and cross-source) — prevents identical files
- ✅ Adaptive JPEG compression to target <200 KB (quality 85→40, step 5)
- ✅ Max dimension 720px resize with `Image.Resampling.LANCZOS`
- ✅ Quality validation: minimum file size, aspect ratio, color diversity
- ✅ Post-dedup auto-repair — renumbers files sequentially, rebuilds checkpoint from disk
- ✅ `rebuild_checkpoint_from_disk()` — reads entire `dataset/` and writes fresh checkpoint
- ✅ `renumber_files()` — sequential renaming without gaps after deletion (two-phase rename)
- ✅ Incremental scraping — resumes from existing dataset (reads hash + pHash from checkpoint)
- ✅ Thread-safe — `threading.Lock` for counter, hash set, and pHash synchronization
- ✅ Automatic checkpoint per keyword (`pexels_checkpoint.json`) — stores SHA256 + pHash
- ✅ Signal handling `SIGTERM/SIGINT` and invalid URL filtering (logos, avatars, icons, SVG, GIF, video, etc.)
- ✅ Interactive CLI with `--stats`, `--recompress`, `--dedup --dry-run`, `--dedup`

### How to Extend the Dataset

**Scenario: Expand from 13,440 (1,680/class) to ~20,000 (2,500/class)**

1. Edit `config_images.json`:
   ```json
   "target_per_class": 2500,
   "total_target": 20000
   ```

2. Run scraper:
   ```bash
   python3 pexels_scraper.py 2500
   ```

3. Behind the scenes:
   - `load_existing()` reads 1,680 files per class from disk → `stats=1680`, `next_idx=1680`
   - `load_checkpoint()` loads 13,440 SHA256 + pHash from checkpoint
   - `should_stop()` checks `1680 < 2500` → continue scraping
   - `save_img()` rejects images with existing SHA256 (exact dedup)
   - `save_img()` rejects images with similar pHash (threshold ≤ 8)
   - New files saved as `bridge_01680.jpg`, `bridge_01681.jpg`, etc.

4. To add a new class, add entry to `classes` array in `config_images.json`:
   ```json
   {
     "name": "cathedral",
     "keywords": ["cathedral architecture", "gothic cathedral", "cathedral building"]
   }
   ```

## How to Load

### Python (HuggingFace datasets)

```python
from datasets import load_dataset

dataset = load_dataset("0xgr3y/arch-building-dataset", data_dir="dataset")
```

### Python (TensorFlow/Keras)

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(320, 320),
    batch_size=32
)
```

### Python (PyTorch)

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("dataset/", transform=transform)
```

## Limitations & Sampling Bias

- **Pexels platform bias:** Images sourced from Pexels, a Western-centric photography platform. This may introduce bias toward Euro-American architectural styles and professional photography aesthetics. Users should evaluate on geographically diverse test sets before deployment.
- **No inter-annotator agreement metric:** Labels were assigned by search keywords (e.g., "mosque architecture") and verified by single-curator human review. Unlike multi-annotator datasets with Cohen's Kappa, this dataset uses keyword-driven collection with human curation, which may introduce curator bias.
- **Resolution cap:** Images compressed to max 720px longest side. Fine-grained architectural details (e.g., window patterns, ornamental carvings) may be lost at this resolution.
- **Single source:** All images from Pexels. No cross-platform validation (e.g., Unsplash, Flickr) to assess source-independent classification accuracy.
- **Barn and windmill share 3 cross-class duplicates** (0.02% of dataset) — left as-is due to negligible impact on training.

## Ethical Considerations

- **No personally identifiable information (PII):** All images are photographs of architectural buildings — no faces, names, or personal data included.
- **Human content filtered:** URL pattern filters explicitly exclude images tagged with people, portraits, or human-focused content during collection.
- **Commercial-free licensing:** All images sourced from Pexels under the Pexels License (free for commercial use, no attribution required).
- **Building-only scope:** This dataset is strictly limited to architectural structures. It does not include content that could be used for surveillance, profiling, or any harmful purpose.
- **Responsible use:** Users agree not to claim ownership of original images, redistribute without attribution, or use the dataset for unlawful purposes.

## Files in This Repository

| File | Description |
|------|-------------|
| `dataset/` | 13,440 images (8 classes × 1,680, JPEG format) |
| `pexels_scraper.py` | Multi-mode scraper (Pexels API + BrightData + Selenium) |
| `config_images.json` | Scraping configuration (classes, keywords, parameters, dedup) |
| `pexels_checkpoint.json` | SHA256 + pHash checkpoint for resume-capable scraping |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Related

- **Model:** [0xgr3y/Arch-Building-Image-Classification](https://huggingface.co/0xgr3y/Arch-Building-Image-Classification) (EfficientNetV2-S, GeMPooling, Focal Loss, DiscriminativeAdamW, SWA)
- **Live Demo:** [0xgr3y/arch-building-classifier](https://huggingface.co/spaces/0xgr3y/arch-building-classifier) (Gradio Space)
- **Model Code & Training:** [arcxteam/building-architectural-image-classifier](https://github.com/arcxteam/building-architectural-image-classifier) (notebook.ipynb, training, evaluation)
- **HuggingFace Dataset:** [0xgr3y/arch-building-dataset](https://huggingface.co/datasets/0xgr3y/arch-building-dataset) (mirror)

## Citation

### APA (7th Edition)

> Saugani. (2026). *World Architectural Buildings Dataset (FGIC) for Multi-Class Image Classification* [Dataset]. GitHub. https://github.com/arcxteam/dataset-fgic-architectural

### BibTeX — GitHub (this repository)

```bibtex
@misc{saugani2026_arch_building_dataset_github,
  title={World Architectural Buildings Dataset (FGIC) for Multi-Class Image Classification},
  author={Saugani},
  year={2026},
  publisher={GitHub},
  license={CC-BY-4.0},
  url={https://github.com/arcxteam/dataset-fgic-architectural}
}
```

### BibTeX — HuggingFace (mirror with DOI)

```bibtex
@misc{saugani2026_arch_building_dataset_hf,
  title={World Architectural Buildings Dataset (FGIC) for Multi-Class Image Classification},
  author={Saugani},
  year={2026},
  publisher={Hugging Face},
  license={CC-BY-4.0},
  url={https://huggingface.co/datasets/0xgr3y/arch-building-dataset},
  doi={10.57967/hf/9220}
}
```

## License

- **Dataset:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Source images:** [Pexels License](https://www.pexels.com/license/) (free for commercial use, no attribution required)
- **Source code:** [MIT License](https://opensource.org/licenses/MIT)
