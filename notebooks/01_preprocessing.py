"""
Sesi 1: Preprocessing - Ekstraksi dan Pembersihan Teks Putusan Pengadilan
Mata Kuliah: Penalaran Komputer B
Domain: Pidana Khusus - Perdagangan Orang
"""

import os
import re
import pdfplumber
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, '..', '..', 'pdf')
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def extract_clean_text_from_pdf(pdf_path):
    """Ekstrak teks dari PDF dengan filter font untuk menghilangkan watermark"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_lines = []
            for page in pdf.pages:
                chars = page.chars
                if not chars:
                    continue

                # Filter: hanya karakter dengan ukuran font antara 6pt dan 40pt
                # (watermark menggunakan Helvetica-Bold >= 45pt)
                real_chars = [
                    c for c in chars
                    if 6.0 <= round(c.get('size', 0), 1) <= 40.0
                ]

                if not real_chars:
                    continue

                # Kelompokkan berdasarkan posisi y (pembulatan)
                y_lines = {}
                for c in real_chars:
                    y_key = round(c['y0'], 0)
                    if y_key not in y_lines:
                        y_lines[y_key] = []
                    y_lines[y_key].append(c)

                # Urutkan dari atas ke bawah, lalu kiri ke kanan
                for y_key in sorted(y_lines, reverse=True):
                    line_chars = sorted(y_lines[y_key], key=lambda c: c['x0'])
                    line_text = ''.join(c['text'] for c in line_chars)
                    all_lines.append(line_text)

            return '\n'.join(all_lines) if all_lines else ''
    except Exception as e:
        print(f"  Error ekstraksi PDF: {e}")
        return None


def post_clean_text(text):
    """Pembersihan lanjutan: hapus header, footer, disclaimer"""
    lines = text.split('\n')
    cleaned = []
    skip_until_empty = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if skip_until_empty:
                skip_until_empty = False
            continue

        # Mulai skip saat menemui Disclaimer
        if 'Disclaimer' in stripped:
            skip_until_empty = True
            continue

        if skip_until_empty:
            continue

        # Hapus header/watermark umum
        if stripped == 'P U T U S A N':
            continue
        if stripped == 'M A H K A M A H     A G U N G' or stripped.startswith('M A H K A M A H'):
            continue
        if 'Direktori Putusan Mahkamah Agung' in stripped:
            continue
        if 'putusan.mahkamahagung.go.id' in stripped:
            continue
        if stripped == 'DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA':
            continue

        # Hapus footer halaman
        if re.match(r'^Halaman \d+ dari \d+ halaman', stripped):
            continue
        if re.match(r'^Halaman \d+$', stripped):
            continue
        if 'Kepaniteraan Mahkamah Agung' in stripped:
            skip_until_empty = True
            continue

        # Hapus email/telepon
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(go\.id|com|org)$', stripped):
            continue
        if re.match(r'^\d{2,3}-\d{3,4}-\d{4}$', stripped):
            continue

        # Hapus baris yang hanya berisi scattered chars dari watermark sisa
        words = stripped.split()
        meaningful = [w for w in words if len(w) > 1 or w.isdigit()]
        if len(meaningful) == 0 and len(words) <= 3:
            continue

        cleaned.append(stripped)

    # Gabungkan dan normalisasi spasi
    result = '\n'.join(cleaned)
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'[ \t]+', ' ', result)

    # Bersihkan karakter tunggal yang tersisa di awal/akhir baris
    lines2 = result.split('\n')
    cleaned2 = []
    for line in lines2:
        # Hapus 1-2 karakter di awal baris yang terisolasi
        line = re.sub(r'^[a-zA-Z]\s+', '', line)
        line = re.sub(r'\s+[a-zA-Z]$', '', line)
        line = re.sub(r'^[a-zA-Z]\s[a-zA-Z]\s+', '', line)
        if line.strip():
            cleaned2.append(line.strip())

    return '\n'.join(cleaned2)


def validate_content(cleaned_text):
    """Validasi bahwa konten memiliki cukup teks bermakna"""
    words = cleaned_text.split()
    meaningful_words = [w for w in words if len(w) > 2]
    ratio = len(meaningful_words) / len(words) if words else 0
    return {
        'total_words': len(words),
        'meaningful_words': len(meaningful_words),
        'ratio': ratio,
        'is_valid': len(words) >= 200 and ratio >= 0.5
    }


def main():
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, '*.pdf')))
    print(f"Total PDF ditemukan: {len(pdf_files)}")

    results = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        case_id = f"case_{idx:03d}"
        pdf_name = os.path.basename(pdf_path)
        print(f"[{idx:2d}/70] {pdf_name}", end=' ')

        # Ekstrak teks dengan filter font
        raw_text = extract_clean_text_from_pdf(pdf_path)

        if not raw_text or len(raw_text.strip()) == 0:
            print("GAGAL - teks kosong")
            continue

        # Pembersihan lanjutan
        clean = post_clean_text(raw_text)

        # Validasi
        validation = validate_content(clean)
        status = "OK" if validation['is_valid'] else "LOW"

        print(f"-> {validation['total_words']} kata, rasio={validation['ratio']:.1%} [{status}]")

        # Simpan
        output_path = os.path.join(RAW_DIR, f"{case_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean)

        results.append({
            'case_id': case_id,
            'file': pdf_name,
            'total_words': validation['total_words'],
            'ratio': validation['ratio'],
            'status': status
        })

        if idx % 10 == 0:
            print(f"  --- {idx} selesai ---")

    print("\n" + "=" * 50)
    print("HASIL PREPROCESSING")
    print("=" * 50)
    ok = [r for r in results if r['status'] == 'OK']
    low = [r for r in results if r['status'] == 'LOW']
    print(f"Berhasil: {len(ok)}/{len(results)}")
    print(f"Peringatan (< 200 kata): {len(low)}")
    for r in low:
        print(f"  - {r['file']}: {r['total_words']} kata")
    print(f"\nFile bersih tersimpan di: {RAW_DIR}")


if __name__ == '__main__':
    main()
