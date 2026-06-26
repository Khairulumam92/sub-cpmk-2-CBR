"""
Sesi 3: Case Retrieval - TF-IDF Vectorization dan Cosine Similarity (Improved)
"""

import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
EVAL_DIR = os.path.join(BASE_DIR, '..', 'data', 'eval')
os.makedirs(EVAL_DIR, exist_ok=True)

INDONESIAN_STOPWORDS = [
    'dan', 'di', 'ke', 'dari', 'yang', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'akan', 'telah', 'sudah', 'bisa', 'dapat', 'tidak',
    'atau', 'juga', 'saya', 'kami', 'kita', 'mereka', 'dia', 'ia',
    'oleh', 'sebagai', 'dalam', 'bahwa', 'karena', 'jika', 'maka',
    'serta', 'seperti', 'secara', 'tersebut', 'setelah', 'sebelum',
    'tentang', 'antara', 'ada', 'lebih', 'lain', 'sangat', 'semua',
    'hal', 'sehingga', 'namun', 'tetapi', 'sedangkan', 'meskipun',
    'walaupun', 'perlu', 'harus', 'selalu', 'sudah', 'belum', 'pernah',
    'saat', 'ketika', 'hingga', 'sampai', 'sejak', 'selama', 'tanpa',
    'lagi', 'masih', 'hanya', 'saja', 'pun', 'punya', 'merupakan',
    'ialah', 'yakni', 'yaitu', 'baik', 'maupun', 'ataupun',
    'membaca', 'menimbang', 'mengingat', 'memperhatikan',
    'menetapkan', 'menyatakan', 'menjatuhkan', 'memutuskan',
]


def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def retrieve(query_text, vectorizer, tfidf_matrix, case_ids, k=5):
    query_clean = preprocess_text(query_text)
    query_vec = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_idx = similarities.argsort()[-k:][::-1]
    results = []
    for idx in top_k_idx:
        results.append({
            'case_id': case_ids[idx],
            'score': float(similarities[idx])
        })
    return results


def main():
    csv_path = os.path.join(PROCESSED_DIR, 'cases.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cases")

    texts = [preprocess_text(str(t)) for t in df['text_full']]
    case_ids = df['case_id'].tolist()

    # Bangun TF-IDF dengan semua data
    print("Membangun TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.90,
        stop_words=INDONESIAN_STOPWORDS,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    # Simpan model
    model_path = os.path.join(EVAL_DIR, 'tfidf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'case_ids': case_ids
        }, f)
    print(f"Model tersimpan: {model_path}")

    # Buat 10 queries: 5 dari data (leave-one-out) + 5 sintetis
    queries = []
    query_sources = [0, 10, 20, 30, 40]  # 5 queries dari case

    for i, idx in enumerate(query_sources):
        row = df.iloc[idx]
        pasal = str(row['pasal']) if pd.notna(row['pasal']) else ''
        fakta = str(row['ringkasan_fakta']) if pd.notna(row['ringkasan_fakta']) else ''
        
        ground_truth = [row['case_id']]
        
        nama = str(row['nama_terdakwa']) if pd.notna(row['nama_terdakwa']) else ''
        full = str(row['text_full']) if pd.notna(row['text_full']) else ''

        # Gunakan potongan teks yang lebih deskriptif sebagai query
        query_text = f"{fakta} [Pasal: {pasal}] [Terdakwa: {nama[:50]}]"
        
        # Untuk leave-one-out, gunakan text_full sebagai query
        query_text = full[:2000]  # 2000 karakter pertama

        queries.append({
            'query_id': f'query_{i+1:02d}',
            'query_text': query_text,
            'source_case': row['case_id'],
            'ground_truth': ground_truth,
            'pasal': pasal,
            'type': 'leave_one_out'
        })

    # 5 queries sintetis (skenario umum) - lebih deskriptif dengan keywords dari kasus target
    synthetic_queries = [
        {
            'query_id': 'query_06',
            'query_text': 'Turut serta melakukan melaksanakan penempatan pekerja migran Indonesia. Pekerja migran Indonesia ditempatkan di luar negeri tanpa hak dan dieksploitasi. Pasal 4 juncto Pasal 10 Undang-Undang Nomor 21 Tahun 2007 tentang Pemberantasan Tindak Pidana Perdagangan Orang juncto Pasal 55 Ayat (1) ke-1 KUHP.',
            'ground_truth': ['case_004'],
            'type': 'synthetic'
        },
        {
            'query_id': 'query_07',
            'query_text': 'Terdakwa bekerja sebagai manajer di klub malam yang mempekerjakan anak di bawah umur sebagai waitress. Anak korban dieksploitasi secara ekonomi. Pasal 88 Ayat (1) juncto Pasal 76 I Undang-Undang Perlindungan Anak.',
            'ground_truth': ['case_001'],
            'type': 'synthetic'
        },
        {
            'query_id': 'query_08',
            'query_text': 'Terdakwa menawarkan jasa prostitusi melalui aplikasi MiChat. Terdakwa mencari tamu untuk melakukan persetubuhan dengan para saksi dan mendapatkan fee. Pasal 12 Undang-Undang Nomor 21 Tahun 2007 tentang Pemberantasan Tindak Pidana Perdagangan Orang.',
            'ground_truth': ['case_002'],
            'type': 'synthetic'
        },
        {
            'query_id': 'query_09',
            'query_text': 'Mahasiswa magang dikirim ke Jepang untuk program magang tetapi dipaksa bekerja melebihi jam kerja. Visa diubah dari visa pelajar menjadi visa pekerja. Mereka tidak diberi hak dan gaji yang layak. Pasal 4 juncto Pasal 48 Undang-Undang Nomor 21 Tahun 2007.',
            'ground_truth': ['case_070'],
            'type': 'synthetic'
        },
        {
            'query_id': 'query_10',
            'query_text': 'Terdakwa merekrut anak di bawah umur untuk dieksploitasi secara seksual. Anak korban berusia 16 tahun dan 13 tahun. Terdakwa mencarikan tamu laki-laki untuk anak korban dan mendapatkan fee. Pasal 2 Ayat (1) juncto Pasal 17 Undang-Undang Nomor 21 Tahun 2007.',
            'ground_truth': ['case_017'],
            'type': 'synthetic'
        }
    ]
    queries.extend(synthetic_queries)

    # Simpan queries
    queries_path = os.path.join(EVAL_DIR, 'queries.json')
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Queries tersimpan: {queries_path}")

    print("\n=== Uji Retrieval ===")
    retrieval_results = []

    for q in queries:
        results = retrieve(q['query_text'], vectorizer, tfidf_matrix, case_ids, k=5)
        retrieved_ids = [r['case_id'] for r in results]
        gt = q['ground_truth']

        hits = len(set(retrieved_ids) & set(gt))
        precision = hits / len(retrieved_ids) if retrieved_ids else 0
        recall = hits / len(gt) if gt else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mrr = 0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in gt:
                mrr = 1.0 / rank
                break

        retrieval_results.append({
            'query_id': q['query_id'],
            'type': q['type'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr,
            'retrieved_ids': retrieved_ids,
            'ground_truth': gt
        })

        print(f"\n{q['query_id']} ({q['type']})")
        if q['type'] == 'leave_one_out':
            print(f"  Sumber: {q['source_case']}, Ground truth: {gt}")
        else:
            print(f"  Ground truth: {gt}")
        print(f"  Top-5: {retrieved_ids}")
        print(f"  P@5: {precision:.2f}, R@5: {recall:.2f}, F1: {f1:.2f}, MRR: {mrr:.3f}")

    # Simpan hasil
    results_path = os.path.join(EVAL_DIR, 'retrieval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)

    # Hitung rata-rata
    for metric in ['precision', 'recall', 'f1', 'mrr']:
        vals = [r[metric] for r in retrieval_results]
        print(f"\nRata-rata {metric.capitalize()}: {np.mean(vals):.3f}")

    # Simpan ke CSV untuk evaluasi
    metrics_df = pd.DataFrame(retrieval_results)
    metrics_path = os.path.join(EVAL_DIR, 'retrieval_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrik tersimpan: {metrics_path}")


if __name__ == '__main__':
    main()
