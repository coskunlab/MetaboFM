# MetaboFM: A Foundation Model for Spatial Metabolomics

> **MetaboFM** unifies large-scale **mass spectrometry imaging (MSI)** curation, **Vision Transformer–based** representation learning, **spatio–spectral interpretability**, and multimodal **visual question answering (VQA)** for interactive and interpretable spatial metabolomics.

---

## 🔑 Highlights

- **Large-scale curated MSI corpus** from METASPACE with consistent FDR filtering, MSM-based channel prioritization, percentile normalization, tiling, and metadata harmonization across six categories.
- **Vision Transformers for MSI** — pretrained **DINOv2–ViT-B/14** and **MAE–ViT-B/16** adapted via a **two-phase multi-task fine-tuning** strategy.
- **Six metadata tasks:** organism, ionization polarity, organ/tissue, condition, analyzer type, ionization source.
- **Linear and few-shot probes** to quantify transferability and label efficiency.
- **Spatio–spectral attention** combines transformer attention rollout with Input×Grad channel saliency to map embedding directions to spatial regions and their most influential **m/z** channels.
- **VQA module** fuses MSI embeddings with text embeddings through a lightweight fusion MLP to answer natural-language metadata questions.
- **Interactive Gradio app** for uploading MSI tiles, visualizing PCA-RGB / single-channel images, and querying metadata using free-form VQA.

---

## 🧪 Key Results

- **Linear probe (frozen encoders):** pretrained DINOv2–ViT-B/14 achieves **macro–F1 = 0.74** across metadata tasks.
- **MSI multi-task fine-tuning** significantly improves separability for organ/tissue, condition, analyzer type, and ionization source.
- **Few-shot learning:** steady performance increases with {1, 5, 10, 25} labeled samples per class.
- **Unsupervised structure:** MSI fine-tuning yields **>2× ARI improvement**.
- **Spatio–spectral attention:** top-ranked **m/z** channels align with meaningful tissue morphology and disease structure.
- **Case study (Healthy vs Tumor kidney):** macro–F1 = **0.86**, with interpretable spectral and spatial markers.
- **VQA:** 5-fold CV performance **macro–F1 = 0.69 ± 0.03**, accuracy **0.79 ± 0.02**.

---

## 🚀 Interactive Demo

🔗 **Try MetaboFM in your browser:**  
https://huggingface.co/spaces/efesthefirst/metabofm

| File | Purpose |
|------|---------|
| `demo.npz` | Example MSI tile for quick demo. Auto-load button included. |

### How to use the demo:
1. Open the Hugging Face Space  
2. Click **Load example demo.npz** or upload your own `.npz`  
3. View PCA-RGB or spectral channels  
4. Ask free-form questions (e.g., *"What is the ionization polarity?"*)  

---

# 📦 Software & Code Information

## 🔧 System Requirements

**Operating systems:**
- Ubuntu 20.04 / 22.04  
- Windows 10 / 11  

**Python:** 3.10  

**Hardware:**
- CPU-compatible  
- GPU recommended (≥8GB VRAM)

**Key dependencies:**
- PyTorch ≥ 2.2  
- timm ≥ 0.9  
- transformers  
- scikit-learn, NumPy, SciPy  
- Gradio  
- tqdm  

Full list in `requirements.txt`.

---

## 🛠️ Installation

git clone https://github.com/coskunlab/MetaboFM.git
cd MetaboFM
pip install -r requirements.txt

## 📁 Notebooks

All results are reproducible via the notebooks in:


### Notebooks include:

- **[`01_metapace.ipynb`](notebooks/01_metapace.ipynb)**  
  — dataset retrieval & curation

- **[`02_create_df_meta.ipynb`](notebooks/02_create_df_meta.ipynb)**  
  — metadata harmonization

- **[`03_multitask_finetuning.ipynb`](notebooks/03_multitask_finetuning.ipynb)**  
  — foundation model training

- **[`04_downstream_tasks.ipynb`](notebooks/04_downstream_tasks.ipynb)**  
  — linear probes, few-shot learning, ARI clustering

- **[`05_ari.ipynb`](notebooks/05_ari.ipynb)**  
  — unsupervised clustering metrics

- **[`06_spatio_spectral_attention.ipynb`](notebooks/06_spatio_spectral_attention.ipynb)**  
  — interpretability module

- **[`07_application.ipynb`](notebooks/07_application.ipynb)**  
  — kidney case study

- **[`08_vqa.ipynb`](notebooks/08_vqa.ipynb)**  
  — VQA training & evaluation

---

## ⚖️ License

MetaboFM is released under the **MIT License**.
