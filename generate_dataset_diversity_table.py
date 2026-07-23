"""
Generate dataset diversity summary table for Supplementary Table 1.

Counts samples per ionisation source, analyzer type, organism, polarity, and
top-15 organ/tissue types. Outputs a CSV suitable for inclusion in the manuscript.

Usage
-----
  python generate_dataset_diversity_table.py

Output
------
  outputs/dataset_diversity_table.csv
"""
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import pandas as pd

CH_META  = METABOFM_ROOT / "outputs/embeddings_v2/stage2_channel_meta.csv"
OUT_PATH = METABOFM_ROOT / "outputs/dataset_diversity_table.csv"

def main():
    df   = pd.read_csv(CH_META)
    samp = df.drop_duplicates("sample_path").copy()

    print(f"Total samples : {len(samp):,}")
    print(f"Total channels: {len(df):,}")

    rows = []

    def add_group(category, col, top_n=None):
        vc = samp[col].value_counts()
        if top_n:
            vc = vc.head(top_n)
        for val, cnt in vc.items():
            rows.append({"Category": category, "Value": val, "N_samples": cnt})
        rows.append({"Category": category, "Value": "TOTAL", "N_samples": len(samp)})
        rows.append({"Category": "", "Value": "", "N_samples": ""})

    add_group("Ionisation source", "ionisationSource")
    add_group("Analyzer type",     "analyzerType")
    add_group("Organism",          "organism")
    add_group("Polarity",          "polarity")

    rows.append({"Category": "Organ / tissue (top 15)", "Value": "(all)", "N_samples": len(samp)})
    for val, cnt in samp["Organism_Part"].value_counts().head(15).items():
        rows.append({"Category": "Organ / tissue", "Value": val, "N_samples": cnt})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(str(OUT_PATH), index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
