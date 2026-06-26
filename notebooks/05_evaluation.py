"""
Sesi 5: Evaluasi Model - Metrik Retrieval dan Prediksi
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, '..', 'data', 'eval')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'data', 'results')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')


def evaluate_retrieval():
    """Evaluasi metrik retrieval"""
    retrieval_path = os.path.join(EVAL_DIR, 'retrieval_results.json')
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    metrics = []
    for r in results:
        retrieved = set(r['retrieved_ids'])
        ground = set(r['ground_truth'])

        true_pos = len(retrieved & ground)
        false_pos = len(retrieved - ground)
        false_neg = len(ground - retrieved)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Mean Reciprocal Rank
        mrr = 0
        for rank, cid in enumerate(r['retrieved_ids'], 1):
            if cid in ground:
                mrr = 1.0 / rank
                break

        metrics.append({
            'query_id': r['query_id'],
            'type': r['type'],
            'precision@5': round(precision, 3),
            'recall@5': round(recall, 3),
            'f1_score': round(f1, 3),
            'mrr': round(mrr, 3),
            'top1_match': 'YES' if mrr > 0 else 'NO'
        })

    df = pd.DataFrame(metrics)
    output_path = os.path.join(EVAL_DIR, 'retrieval_metrics.csv')
    df.to_csv(output_path, index=False)

    print("=== METRIK RETRIEVAL ===")
    print(df.to_string(index=False))

    print(f"\nRata-rata:")
    for col in ['precision@5', 'recall@5', 'f1_score', 'mrr']:
        print(f"  {col}: {df[col].mean():.3f}")
    print(f"  Query dengan top-1 match: {df['top1_match'].value_counts().get('YES', 0)}/{len(df)}")

    return df


def evaluate_prediction():
    """Evaluasi metrik prediksi"""
    pred_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    df = pd.read_csv(pred_path)

    print("\n=== METRIK PREDIKSI ===")

    # Filter hanya queries dengan ground truth > 0
    valid = df[df['ground_truth_pidana'] > 0].copy()

    if valid.empty:
        print("Tidak ada data valid untuk evaluasi prediksi")
        return df

    # Mean Absolute Error
    mae_pidana = mean_absolute_error(valid['ground_truth_pidana'], valid['predicted_pidana'])
    mae_denda = mean_absolute_error(valid['ground_truth_denda'], valid['predicted_denda'])

    print(f"MAE Pidana Penjara: {mae_pidana:.2f} tahun")
    print(f"MAE Denda: Rp{mae_denda:,.0f}")

    # Kategorisasi error
    valid['error_pidana'] = abs(valid['predicted_pidana'] - valid['ground_truth_pidana'])
    valid['error_denda'] = abs(valid['predicted_denda'] - valid['ground_truth_denda'])

    print(f"\nError Analysis:")
    print(f"  Error pidana <= 2 tahun: {(valid['error_pidana'] <= 2).sum()}/{len(valid)}")
    print(f"  Error pidana <= 5 tahun: {(valid['error_pidana'] <= 5).sum()}/{len(valid)}")

    # Simpan metrik prediksi
    metrics = valid[['query_id', 'type', 'predicted_pidana', 'ground_truth_pidana',
                     'predicted_denda', 'ground_truth_denda', 'error_pidana', 'error_denda']].copy()
    metrics_path = os.path.join(EVAL_DIR, 'prediction_metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrik prediksi tersimpan: {metrics_path}")

    # Analisis kegagalan (rejection analysis)
    print("\n=== ANALISIS KEGAGALAN ===")
    worst = valid.nlargest(3, 'error_pidana')
    for _, row in worst.iterrows():
        print(f"\n{row['query_id']} ({row['type']})")
        print(f"  Prediksi: {row['predicted_pidana']} thn, Ground: {row['ground_truth_pidana']} thn")
        print(f"  Error: {row['error_pidana']:.1f} tahun")
        print(f"  Top-5: {row['top_5_case_ids']}")

    # Analisis mengapa gagal
    print("\nPenyebab potensial kegagalan:")
    print("1. Ringkasan fakta query tidak memiliki cukup kesamaan dengan kasus target")
    print("2. Kasus target memiliki pasal yang unik/berbeda dari retrieved cases")
    print("3. Terdapat noise pada ekstraksi pidana dari teks (parsing error)")
    print("4. Beberapa kasus memiliki pidana 0 (bebas) yang mempengaruhi rata-rata")

    return df


def main():
    print("EVALUASI SISTEM CBR - ANALISIS PUTUSAN PENGADILAN")
    print("=" * 50)

    ret_metrics = evaluate_retrieval()
    print("\n")
    pred_metrics = evaluate_prediction()

    print("\n" + "=" * 50)
    print("RINGKASAN EVALUASI")
    print("=" * 50)
    print(f"Total kasus dalam case base: 70")
    print(f"Total query uji: 10")
    print(f"  - Leave-one-out: 5")
    print(f"  - Skenario sintetis: 5")

    avg_prec = ret_metrics['precision@5'].mean()
    avg_rec = ret_metrics['recall@5'].mean()
    avg_f1 = ret_metrics['f1_score'].mean()
    avg_mrr = ret_metrics['mrr'].mean()

    print(f"\nRetrieval:")
    print(f"  Precision@5: {avg_prec:.3f}")
    print(f"  Recall@5:    {avg_rec:.3f}")
    print(f"  F1-Score:    {avg_f1:.3f}")
    print(f"  MRR:         {avg_mrr:.3f}")

    # MAE from prediction
    pred_csv = os.path.join(RESULTS_DIR, 'predictions.csv')
    pred_df = pd.read_csv(pred_csv)
    valid_pred = pred_df[pred_df['ground_truth_pidana'] > 0]
    if not valid_pred.empty:
        mae_p = mean_absolute_error(valid_pred['ground_truth_pidana'], valid_pred['predicted_pidana'])
        mae_d = mean_absolute_error(valid_pred['ground_truth_denda'], valid_pred['predicted_denda'])
        print(f"\nPrediksi:")
        print(f"  MAE Pidana: {mae_p:.2f} tahun")
        print(f"  MAE Denda: Rp{mae_d:,.0f}")


if __name__ == '__main__':
    main()
