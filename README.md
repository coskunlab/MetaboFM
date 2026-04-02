# MetaboFM: Multimodal Representation Learning for Spatial Metabolomics

> **MetaboFM** is a multimodal representation learning framework for **mass spectrometry imaging (MSI)** that integrates **MSI-derived spatial embeddings** with **molecular structure information encoded from SMILES strings** to learn transferable representations across large and heterogeneous spatial metabolomics datasets.

---

## 🔑 Highlights

- **Large-scale curated MSI corpus** aggregated primarily from **METASPACE**, spanning diverse organs, conditions, ionization sources, analyzer types, and polarities.
- **Masked autoencoder (MAE)-based representation learning** for stacked MSI ion images using a Vision Transformer backbone.
- **Channel-aware training strategies**, including channel permutation, channel dropout, and structured channel retention, to improve robustness to sparse and heterogeneous MSI measurements.
- **Two complementary MSI representations**:
  - **Sample-level embeddings** capturing global spatial metabolite organization
  - **Channel-level embeddings** capturing annotation-level metabolite-associated spatial patterns
- **Multimodal fusion** of MSI embeddings with **SMILES-derived molecular structure embeddings**.
- **Benchmark evaluation suite** covering:
  - biological classification
  - metabolite-level semantic prediction
  - nearest-neighbor retrieval
  - molecule-level embedding analysis
- **Context-aware molecular representation**, showing that repeated observations of the same molecular annotation can shift systematically across tissue environments rather than collapsing to a single invariant representation.

---

## 🧪 Key Results

- Trained on approximately:
  - **5,800 MSI datasets**
  - **165,000 ion images**
  - **27,000 distinct molecules**
- Learned embeddings consistently outperformed MSI-only and engineered baselines across benchmark tasks.
- For metabolite-level semantic prediction, the best-performing multimodal representation achieved:
  - **Macro-F1 = 0.773 ± 0.013**
  - **Macro-F1 = 0.783 ± 0.010**
  on hierarchical metabolite annotation tasks.
- The learned embedding space preserved:
  - **biological variation**, supporting condition and organ classification
  - **chemical structure**, supporting metabolite-level semantic organization and retrieval
  - **context-dependent molecular variation**, where the same annotation changes representation according to tissue environment

---

## 📘 Overview

Mass spectrometry imaging enables label-free molecular mapping across tissues, but MSI datasets are high-dimensional, sparse, and highly heterogeneous across biological systems and acquisition platforms. MetaboFM addresses this challenge by learning a unified embedding space that combines:

- **spatial information** from MSI ion images
- **chemical structure information** from molecular SMILES strings
- **biological and technical metadata** associated with public MSI datasets

The framework is designed to support transferable analysis across diverse MSI datasets and downstream tasks, including biological prediction, molecular retrieval, and metabolite-level semantic annotation.

---

## 🧠 Method Summary

MetaboFM consists of five main stages:

1. **MSI dataset curation and preprocessing**
   Public MSI datasets are aggregated and harmonized. Retained annotations are filtered and ranked by confidence, and top ion images are stacked into multi-channel MSI inputs.

2. **Self-supervised MSI representation learning**
   A **masked autoencoder (MAE)** is trained on stacked ion images using channel-aware perturbations, including channel permutation, channel dropout, and structured channel retention.

3. **Embedding extraction**
   The trained encoder produces:
   - **sample-level embeddings** from full multi-channel MSI inputs
   - **channel-level embeddings** from individual ion images

4. **Molecular structure integration**
   Candidate molecules with valid PubChem-derived SMILES strings are encoded with **Molformer**, producing structure-aware embeddings.

5. **Multimodal fusion and downstream evaluation**
   MSI embeddings and SMILES embeddings are combined in multiple fusion variants and evaluated on classification, retrieval, and semantic prediction tasks.

---

## 📊 Data Sources

MetaboFM integrates MSI data and external molecular knowledge from:

- **METASPACE** — public spatial metabolomics datasets
- **PubChem** — molecular identifiers and structures
- **HMDB** — metabolite taxonomy and hierarchical labels
- **PathBank** — pathway-level biological knowledge

These resources enable joint modeling of spatial ion distributions and molecular annotations across diverse experimental settings.

---

## 📦 Software & Code Information

### 🔧 System Requirements

**Operating systems:**
- Windows 10 / 11

**Python:**
- > Python 3.10

**Hardware:**
- CPU-compatible
- GPU recommended for representation learning and large-scale embedding extraction
- Development and main experiments were run on an **NVIDIA GeForce RTX 4090 GPU**

---

### 🔧 Key Dependencies

- PyTorch
- Hugging Face Transformers
- timm
- scikit-learn
- NumPy
- SciPy
- pandas
- matplotlib
- tqdm

All dependencies are fully specified in **`environment.yaml`**.

---

### 🛠️ Installation Guide

**Typical installation time:** ~10–15 minutes on a normal desktop computer using conda
```bash
git clone https://github.com/coskunlab/MetaboFM.git
cd MetaboFM
conda env create -f environment.yaml
conda activate metabofm
```

---

## ▶️ Running the Code

After installation, the repository can be used to reproduce the major stages of the MetaboFM workflow:

- MSI dataset retrieval and curation
- metadata harmonization
- MSI preprocessing and channel construction
- MAE-based representation learning
- embedding extraction
- multimodal fusion with molecular structure embeddings
- downstream benchmarking and analysis

The primary workflow is organized through notebooks and supporting scripts in the repository.

---

## 📁 Repository Structure
```
MetaboFM/
├── notebooks/
│   ├── 01_metaspace.ipynb
│   ├── 02_metaspace_add_candidate_molecules.ipynb
│   ├── 03_molformer_embed_smiles.ipynb
│   ├── 04_process_knowledgebase.ipynb
│   ├── 05_align_embedding_modalities.ipynb
│   ├── 06_interactive_exploration.ipynb
│   ├── 07_benchmarks.ipynb
│   ├── 08_benchmarks_hmdb.ipynb
│   ├── 09_domain_shift.ipynb
│   ├── 10_interpretability.ipynb
├── environment.yaml
├── README.md
```

---

## 📓 Notebooks

All major analyses are reproducible through the notebooks in the repository:

| Notebook | Description |
|---|---|
| `01_metaspace.ipynb` | Dataset retrieval and curation from public MSI resources |
| `02_metaspace_add_candidate_molecules.ipynb` | Adding and linking candidate molecules to MSI datasets |
| `03_molformer_embed_smiles.ipynb` | Encoding molecular structures from SMILES strings using Molformer |
| `04_process_knowledgebase.ipynb` | Processing and harmonizing metabolite knowledge base resources |
| `05_align_embedding_modalities.ipynb` | Multimodal alignment of MSI and molecular structure embeddings |
| `06_interactive_exploration.ipynb` | Interactive visualization and exploration of the embedding space |
| `07_benchmarks.ipynb` | Biological classification and retrieval benchmark evaluations |
| `08_benchmarks_hmdb.ipynb` | HMDB-based metabolite-level semantic prediction benchmarks |
| `09_domain_shift.ipynb` | Domain shift and cross-dataset generalization analyses |
| `10_interpretability.ipynb` | Embedding interpretation of molecules and spatial correlation analysis |

---

## 📈 Outputs and Reproducibility

The repository is intended to reproduce the main components of the manuscript, including:

- Large-scale MSI preprocessing and annotation filtering
- MAE-based MSI embedding learning
- Extraction of sample-level and channel-level representations
- Fusion with SMILES-derived molecular structure embeddings
- Biological classification benchmarks
- HMDB semantic prediction benchmarks
- Nearest-neighbor retrieval analyses
- Molecule-level embedding visualizations and downstream analyses

For reproducibility:

- Randomized procedures use a fixed seed of **6740**
- Dataset-level splits are used for evaluation
- Downstream classical machine learning analyses are implemented in **scikit-learn**
- Deep learning components are implemented in **PyTorch** using **Hugging Face Transformers**

---

## ⏱️ Expected Runtime

Runtime depends on dataset size, hardware, and whether the user is reproducing full pretraining or only downstream analyses.

| Stage | Approximate Time |
|---|---|
| Environment setup | 10–15 minutes |
| Notebook-level downstream analyses | Minutes to hours depending on dataset size |
| Full large-scale representation learning / embedding extraction | Substantially longer; GPU recommended |

---

## 🧾 Code Availability

All code for the following components is available in the MetaboFM repository:

- Data curation
- Feature extraction
- Representation learning
- Molecular structure integration
- Downstream benchmarking
- Embedding analysis

---

## ⚖️ License

MetaboFM is released under the **MIT License**.
