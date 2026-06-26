# Sistem Case-Based Reasoning (CBR) untuk Analisis Putusan Pengadilan

**Mata Kuliah:** Penalaran Komputer B (Semester Genap 2025/2026)

## Identitas Tim

| NIM | Nama |
|-----|------|
| 202310370311448 | Moh. Khairul Umam |
| 202310370311321 | Nisrina Nurhafizhah |

## Domain

Pidana Khusus - Tindak Pidana Perdagangan Orang (TPPO) berdasarkan Undang-Undang Nomor 21 Tahun 2007.

**Sumber Data:** 70 Dokumen Putusan Mahkamah Agung RI (Format PDF)

## Struktur Direktori

```
.
├── data/
│   ├── raw/            # Teks mentah hasil ekstraksi PDF (case_001.txt - case_070.txt)
│   ├── processed/      # Data terstruktur (cases.csv)
│   ├── eval/           # Query uji, ground truth, dan metrik evaluasi
│   └── results/        # Hasil prediksi solusi
├── notebooks/
│   ├── cbr_pipeline.ipynb           # Notebook utama (seluruh siklus CBR)
│   └── generate_journal.py          # Script generate artikel & presentasi
├── output/                           # Output jurnal & presentasi
│   ├── artikel_joiv.docx            # Artikel jurnal format JOIV
│   ├── presentasi.pptx             # Slide presentasi (5 slide)
│   └── fig*.png                     # Visualisasi (8 gambar)
├── requirements.txt
└── README.md
```

## Cara Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

```bash
# 1. Jalankan pipeline CBR (preprocessing hingga evaluasi)
jupyter notebook notebooks/cbr_pipeline.ipynb

# 2. Generate artikel jurnal & presentasi
python notebooks/generate_journal.py
```

Output akan tersimpan di folder `output/`:
- `artikel_joiv.docx` — Artikel jurnal format JOIV (8-10 halaman)
- `presentasi.pptx` — Slide presentasi (5 slide)

## Metode yang Digunakan

- **Case Retrieval:** 3 model dibandingkan:
  1. TF-IDF + Cosine Similarity (baseline)
  2. TF-IDF + SVM (Support Vector Machine)
  3. TF-IDF + Naive Bayes
- **Train/Test Split:** 80% train / 20% test
- **Case Reuse:** Weighted Similarity Voting
- **Evaluasi:** Precision@5, Recall@5, F1-Score, MRR, MAE, Akurasi Klasifikasi, Visualisasi Bar Chart

## Hasil Evaluasi

### Retrieval (10 query uji)

| Model | Precision@5 | Recall@5 | F1-Score | MRR |
|-------|-------------|----------|----------|-----|
| TF-IDF + Cosine | - | - | - | - |
| TF-IDF + SVM | - | - | - | - |
| TF-IDF + Naive Bayes | - | - | - | - |

### Klasifikasi (TF-IDF)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM | - | - |
| Naive Bayes | - | - |

### Prediksi (Weighted Voting)

| Metrik | Nilai |
|--------|-------|
| MAE Pidana | - tahun |
| MAE Denda | Rp- |

> *Nilai diisi otomatis setelah notebook dijalankan*

## Lisensi

Proyek ini dibuat untuk kepentingan akademis.
