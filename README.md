# MetaboFM: A Foundation Model for Spatial Metabolomics

> **MetaboFM** unifies large-scale **mass spectrometry imaging (MSI)** curation, **Vision Transformer–based** representation learning, **spatio–spectral interpretability**, and multimodal **visual question answering (VQA)** for interactive and interpretable spatial metabolomics.

---

## 🔑 Highlights

- **Large-scale curated MSI corpus** from METASPACE with consistent FDR filtering, MSM-based channel prioritization, percentile normalization, tiling, and metadata harmonization across six categories.
- **Vision Transformers for MSI** — pretrained **DINOv2–ViT-B/14** and **MAE–ViT-B/16** adapted via a **two-phase multi-task fine-tuning** strategy.
- **Six metadata tasks:** organism, ionization polarity, organ/tissue, condition, analyzer type, ionization source.
- **Linear and few-shot probes** to quantify transferability and label efficiency.
- **Spatio–spectral attention** combines transformer attention rollout with Input×Grad channel saliency to map embedding directions to spatial regions and their most influential **m/z** channels.
- **VQA module** fuses MSI embeddings with text embeddings via a **lightweight fusion MLP** to answer natural-language metadata questions.
- **Interactive Gradio app** for uploading MSI tiles, visualizing PCA-RGB / single-channel images, and querying metadata via free-form text prompts.

---

## 🧪 Key Results

- **Linear probe (frozen encoders):** pretrained DINOv2–ViT-B/14 achieves **macro–F1 = 0.74** (mean across six metadata tasks).
- **MSI-specific multi-task fine-tuning** improves discriminative power across tasks, especially for **organ/tissue**, **condition**, **analyzer type**, and **ionization source**.
- **Few-shot learning:** performance increases steadily with {1, 5, 10, 25} labeled samples per class, with fine-tuned DINOv2 consistently outperforming the pretrained variant.
- **Unsupervised structure:** MSI fine-tuning yields **up to >2× improvement in ARI**, indicating sharper metadata-aligned clustering.
- **Spatio–spectral attention:** top-ranked **m/z** channels show spatial patterns consistent with biological / pathological tissue structure.
- **Healthy vs Tumor case study:** MetaboFM embeddings achieve **macro–F1 = 0.86** and identify condition-specific high-attribution **m/z** peaks and spatial saliency in kidney MSI.
- **VQA (5-fold CV):** **macro–F1 = 0.69 ± 0.03**, **accuracy = 0.79 ± 0.02** across six metadata categories using a frozen MSI encoder and partially fine-tuned MiniLM text encoder.
- 
---

## 🚀 Interactive Demo

This repository contains the Gradio interface for MetaboFM:

| File | Purpose |
|------|---------|
| `gradio_app.ipynb` | Upload an MSI tile (`.npz`), visualize it, and ask free-form metadata questions using the MetaboFM VQA model. |

To run the notebook, download the pretrained model checkpoint from Hugging Face:

🔗 https://huggingface.co/efesthefirst/metabofm  
📂 Folder: `20251113_182023/`  
📌 Available weights: `best.pt` (recommended) or `last.pt`
