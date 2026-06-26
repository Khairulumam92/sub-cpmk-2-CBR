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
├── notebooks/          # Script per tahap siklus CBR
│   ├── 01_preprocessing.py
│   ├── 02_representation.py
│   ├── 03_retrieval.py
│   ├── 04_reuse.py
│   └── 05_evaluation.py
├── requirements.txt
└── README.md
```

## Cara Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Pipeline

Jalankan script secara berurutan sesuai siklus CBR:

```bash
cd notebooks

# 1. Preprocessing: Ekstraksi dan pembersihan teks PDF
python 01_preprocessing.py

# 2. Case Representation: Ekstraksi metadata dan fitur
python 02_representation.py

# 3. Case Retrieval: TF-IDF dan cosine similarity
python 03_retrieval.py

# 4. Solution Reuse: Prediksi hasil putusan
python 04_reuse.py

# 5. Evaluasi: Metrik retrieval dan prediksi
python 05_evaluation.py
```

## Metode yang Digunakan

- **Case Retrieval:** TF-IDF Vectorization + Cosine Similarity
- **Case Reuse:** Weighted Similarity Voting
- **Evaluasi:** Precision@5, Recall@5, F1-Score, MRR, MAE

## Hasil Evaluasi

| Metrik | Nilai |
|--------|-------|
| Precision@5 | 0.180 |
| Recall@5 | 0.900 |
| F1-Score | 0.300 |
| MRR | 0.833 |
| MAE Pidana | 3.05 tahun |
| MAE Denda | Rp84,018,800 |

## Lisensi

Proyek ini dibuat untuk kepentingan akademis.
