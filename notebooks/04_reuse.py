"""
Sesi 4: Case Solution Reuse - Prediksi Hasil Putusan
"""

import os
import json
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
EVAL_DIR = os.path.join(BASE_DIR, '..', 'data', 'eval')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(eval_dir):
    with open(os.path.join(eval_dir, 'tfidf_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model['vectorizer'], model['tfidf_matrix'], model['case_ids']


def preprocess_text(text):
    import re
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def retrieve(query, vectorizer, tfidf_matrix, case_ids, k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    query_clean = preprocess_text(query)
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


def predict_outcome(query, vectorizer, tfidf_matrix, case_ids, df, k=5):
    """
    Prediksi solusi menggunakan Weighted Similarity Voting
    """
    results = retrieve(query, vectorizer, tfidf_matrix, case_ids, k)

    total_weight = sum(r['score'] for r in results)
    if total_weight == 0:
        return {
            'predicted_pidana': 0,
            'predicted_denda': 0,
            'top_k_cases': [r['case_id'] for r in results],
            'top_k_scores': [r['score'] for r in results],
            'method': 'weighted_voting'
        }

    weighted_pidana = 0.0
    weighted_denda = 0.0

    for r in results:
        case_data = df[df['case_id'] == r['case_id']]
        if case_data.empty:
            continue
        row = case_data.iloc[0]
        weight = r['score'] / total_weight
        weighted_pidana += weight * float(row['pidana_penjara_tahun'])
        weighted_denda += weight * float(row['denda_rp'])

    return {
        'predicted_pidana': round(weighted_pidana, 1),
        'predicted_denda': round(weighted_denda, -3),  # Dibulatkan ke ribuan
        'top_k_cases': [r['case_id'] for r in results],
        'top_k_scores': [r['score'] for r in results],
        'method': 'weighted_voting'
    }


def main():
    # Load data
    csv_path = os.path.join(PROCESSED_DIR, 'cases.csv')
    df = pd.read_csv(csv_path)

    queries_path = os.path.join(EVAL_DIR, 'queries.json')
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    vectorizer, tfidf_matrix, case_ids = load_model(EVAL_DIR)
    print(f"Model loaded. {len(case_ids)} cases in corpus.")

    print(f"\n=== Prediksi untuk {len(queries)} Queries ===\n")

    predictions = []
    for q in queries:
        qid = q['query_id']
        result = predict_outcome(q['query_text'], vectorizer, tfidf_matrix, case_ids, df, k=5)

        # Cari ground truth dari source_case (untuk leave-one-out queries)
        ground_pidana = 0
        ground_denda = 0
        if q['type'] == 'leave_one_out':
            source = q['source_case']
            src = df[df['case_id'] == source]
            if not src.empty:
                ground_pidana = int(src.iloc[0]['pidana_penjara_tahun'])
                ground_denda = int(src.iloc[0]['denda_rp'])
        else:
            gt = q['ground_truth']
            if gt:
                gt_data = df[df['case_id'].isin(gt)]
                if not gt_data.empty:
                    ground_pidana = int(gt_data.iloc[0]['pidana_penjara_tahun'])
                    ground_denda = int(gt_data.iloc[0]['denda_rp'])

        predictions.append({
            'query_id': qid,
            'type': q['type'],
            'predicted_pidana': result['predicted_pidana'],
            'predicted_denda': result['predicted_denda'],
            'ground_truth_pidana': ground_pidana,
            'ground_truth_denda': ground_denda,
            'top_5_case_ids': '; '.join(result['top_k_cases']),
            'top_5_scores': '; '.join([f"{s:.4f}" for s in result['top_k_scores']]),
            'method': result['method']
        })

        print(f"{qid} ({q['type']})")
        print(f"  Prediksi: {result['predicted_pidana']} thn, Rp{result['predicted_denda']:,.0f}")
        print(f"  Ground:   {ground_pidana} thn, Rp{ground_denda:,.0f}")
        print(f"  Top-5: {result['top_k_cases']}")
        print(f"  Skor: {[round(s, 4) for s in result['top_k_scores']]}")
        print()

    # Simpan hasil
    pred_df = pd.DataFrame(predictions)
    output_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Prediksi tersimpan: {output_path}")

    # Hitung MAE (Mean Absolute Error) untuk pidana
    valid = pred_df[pred_df['ground_truth_pidana'] > 0]
    if not valid.empty:
        mae_pidana = abs(valid['predicted_pidana'] - valid['ground_truth_pidana']).mean()
        print(f"\nMAE Pidana: {mae_pidana:.2f} tahun")
        mae_denda = abs(valid['predicted_denda'] - valid['ground_truth_denda']).mean()
        print(f"MAE Denda: Rp{mae_denda:,.0f}")

    return pred_df


if __name__ == '__main__':
    main()
