"""
Sesi 2: Case Representation - Ekstraksi Metadata dan Fitur
Mata Kuliah: Penalaran Komputer B
"""

import os
import re
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def parse_nomor_perkara(text):
    m = re.search(r'Nomor\s+([\d]+/[A-Za-z/.]+/\d{4})', text)
    return m.group(1) if m else ''


def parse_nama_terdakwa(text):
    lines = text.split('\n')
    nama_lines = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Nama :') or stripped.startswith('Nama:'):
            capture = True
            nama_lines.append(stripped)
        elif capture and stripped.startswith('Tempat Lahir'):
            break
        elif capture and stripped:
            nama_lines.append(stripped)
        elif capture and not stripped:
            break
    return ' '.join(nama_lines).replace('Nama :', '').replace('Nama:', '').strip()


def parse_pasal(text):
    pasal_pattern = re.findall(
        r'Pasal\s+(\d+(?:\s+Ayat\s+\(\d+\))?(?:\s+(?:juncto|dan|atau)\s+Pasal\s+\d+(?:\s+Ayat\s+\(\d+\))?)*)',
        text, re.IGNORECASE
    )
    unique_pasal = []
    for p in pasal_pattern:
        normalized = re.sub(r'\s+', ' ', p.strip())
        if normalized not in unique_pasal:
            unique_pasal.append(normalized)
    return '; '.join(unique_pasal[:10]) if unique_pasal else ''


def parse_pihak(text):
    jaksa = ''
    terdakwa = ''
    m_jaksa = re.search(r'Penuntut Umum pada (Kejaksaan[^;]+)', text)
    if m_jaksa:
        jaksa = m_jaksa.group(1).strip()

    m_terdakwa = re.search(r'Nama\s*:\s*([^;]+)', text)
    if m_terdakwa:
        terdakwa = m_terdakwa.group(1).strip()
    return f"Jaksa: {jaksa} | Terdakwa: {terdakwa}"


def parse_ringkasan_fakta(text):
    """Ambil bagian dakwaan"""
    start = text.find('didakwa dengan dakwaan sebagai berikut:')
    if start == -1:
        start = text.find('dakwaan sebagai berikut:')
    if start == -1:
        start = text.find('Dakwaan:')

    end = text.find('Membaca Tuntutan Pidana')
    if end == -1:
        end = text.find('Membaca Putusan')
    if end == -1:
        end = text.find('Mahkamah Agung tersebut;')

    if start != -1 and end != -1 and end > start:
        return text[start:end].strip()
    return ''


def parse_ringkasan_putusan(text):
    """Ambil amar putusan bagian akhir"""
    markers = ['G A D I L I:', 'G A D I L I;', 'M E N G A D I L I:', 'MENGADILI:', 'G A D I L I']
    found = -1
    for marker in markers:
        pos = text.rfind(marker)
        if pos > found:
            found = pos

    if found == -1:
        return ''

    end_markers = ['Demikianlah diputuskan', 'Demikian diputuskan']
    end_pos = len(text)
    for m in end_markers:
        pos = text.find(m, found)
        if pos != -1 and pos < end_pos:
            end_pos = pos

    return text[found:end_pos].strip()


def parse_pidana_dari_teks(text, amar_text):
    """Ekstrak pidana penjara (tahun) dan denda (Rp) dari amar & fallback teks penuh"""
    pidana_tahun = 0
    denda_rp = 0

    # Priority 1: Cari di amar putusan (GADILI)
    if amar_text:
        m = re.search(r'pidana penjara (?:selama )?(\d+)', amar_text, re.IGNORECASE)
        if m:
            pidana_tahun = int(m.group(1))

    # Priority 2: Jika tidak ditemukan di amar, cari di tuntutan/putusan (0-30 tahun wajar)
    if pidana_tahun == 0:
        pidana_candidates = re.findall(r'pidana penjara (?:selama )?(\d+)', text, re.IGNORECASE)
        for p in pidana_candidates:
            val = int(p)
            if 1 <= val <= 80:
                pidana_tahun = val
                break

    # Denda: cari pola "denda ... RpX" (targeted)
    denda_pattern = re.finditer(
        r'denda\s+(?:terhadap|sebesar|masing-masing)?\s*(?:sebesar\s+)?\s*Rp([\d.,]+)',
        text, re.IGNORECASE
    )
    for m in denda_pattern:
        d_str = m.group(1)
        parts = d_str.split(',')
        integer_part = parts[0].replace('.', '')
        try:
            val = int(integer_part)
            if 50000000 <= val <= 50000000000:
                denda_rp = max(denda_rp, val)
        except ValueError:
            pass

    return pidana_tahun, denda_rp


def parse_tanggal(text):
    """Ekstrak tanggal putusan"""
    m = re.search(r'(?:hari\s+\w+,\s+)?tanggal\s+(\d{1,2}\s+\w+\s+\d{4})', text)
    if m:
        return m.group(1)
    m2 = re.search(r'(\d{1,2}\s+\w+\s+\d{4})\s*telah\s*', text)
    if m2:
        return m2.group(1)
    return ''


def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, 'case_*.txt')))
    print(f"Total file teks ditemukan: {len(raw_files)}")

    cases = []
    for idx, filepath in enumerate(raw_files, 1):
        case_id = f"case_{idx:03d}"
        print(f"[{idx:2d}/70] {case_id}")

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        no_perkara = parse_nomor_perkara(text)
        nama_terdakwa = parse_nama_terdakwa(text)
        tanggal = parse_tanggal(text)
        pasal = parse_pasal(text)
        pihak = parse_pihak(text)
        ringkasan_fakta = parse_ringkasan_fakta(text)
        ringkasan_putusan = parse_ringkasan_putusan(text)
        pidana_tahun, denda_rp = parse_pidana_dari_teks(text, ringkasan_putusan)
        word_count = len(text.split())

        cases.append({
            'case_id': case_id,
            'no_perkara': no_perkara,
            'tanggal': tanggal,
            'nama_terdakwa': nama_terdakwa,
            'ringkasan_fakta': ringkasan_fakta,
            'ringkasan_putusan': ringkasan_putusan,
            'pasal': pasal,
            'pihak': pihak,
            'pidana_penjara_tahun': pidana_tahun,
            'denda_rp': denda_rp,
            'word_count': word_count,
            'text_full': text
        })

        if idx % 10 == 0:
            print(f"  --- {idx} selesai ---")

    df = pd.DataFrame(cases)
    output_path = os.path.join(PROCESSED_DIR, 'cases.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nTersimpan: {output_path}")
    print(f"Total kasus: {len(df)}")
    print(f"Kolom: {list(df.columns)}")

    print("\n=== Ringkasan ===")
    print(f"Rata-rata panjang teks: {df['word_count'].mean():.0f} kata")
    print(f"Rata-rata pidana: {df['pidana_penjara_tahun'].mean():.1f} tahun")
    print(f"Range pidana: {df['pidana_penjara_tahun'].min()} - {df['pidana_penjara_tahun'].max()} tahun")
    print(f"Kasus dengan denda: {df[df['denda_rp'] > 0].shape[0]}/{len(df)}")


if __name__ == '__main__':
    main()
