# MetaboFM: A Foundation Model for Spatial Metabolomics

> **MetaboFM** unifies large-scale **mass spectrometry imaging (MSI)** data curation, pretrained **Vision Transformer–based** representation learning, **m/z** spectral interpretability, and multimodal **visual question answering (VQA)** for interactive spatial metabolomics.

<p align="center">
  <img src="Figures/Figure1.png" width="75%" alt="MetaboFM overview">
</p>

---

## 🔑 Highlights

- **Unified data curation** for public MSI datasets (METASPACE) with consistent FDR, MSM prioritization, tiling, and metadata harmonization.  
- **Pretrained ViTs** (DINOv2, DeiT, MAE) used as frozen feature extractors for MSI—no fine-tuning required.  
- **Six metadata tasks:** organism, ionization polarity, organ/tissue, condition, analyzer type, ionization source.  
- **Linear probes and few-shot probes** to quantify transferability and label efficiency.  
- **m/z attribution** maps latent directions back to spectral intervals for interpretability.  
- **VQA module** integrates MSI embeddings and text encoder via cross-attention for natural-language queries.  
- **Gradio app** for uploading MSI patches, visualizing PCA-RGB/single-channel, and querying metadata.

---

## 🧪 Key Results (from manuscript)

- **Linear probe (frozen encoders):** macro–F1 **0.74**, accuracy **0.80** (mean across six tasks) using **DINOv2–ViT–B/14 (LVD-142M)**. PCA/random baselines trail by **>20 points**.  
- **Task-level bests (macro–F1):** analyzer type 0.80, polarity 0.82, ionization source 0.78, organ/tissue 0.69, organism 0.82, condition 0.64.  
- **VQA (5-fold CV):** macro–F1 0.61 ± 0.05, accuracy 0.74 ± 0.03.  
- **m/z importance:** distinct spectral intervals drive separation for organ/tissue, condition, and ionization source.

---
