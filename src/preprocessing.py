import re
import string

# Indonesian common stopwords (kata hubung dan kata tugas)
# Catatan: Kata negasi seperti "tidak" sengaja TIDAK dimasukkan ke stopwords
# agar polaritas sentimen tidak terbalik.
INDONESIAN_STOPWORDS = set([
    "dan", "yang", "di", "ke", "dari", "ini", "itu", "pada", "adalah", "sebagai",
    "dengan", "untuk", "oleh", "atau", "juga", "sudah", "akan", "telah", "bisa",
    "ada", "saja", "saya", "kami", "mereka", "anda", "dia", "kita", "bukan", "hanya",
    "ia", "pun", "agar", "supaya", "hal", "saat", "karena", "maka", "jika"
])

# Kamus Normalisasi Kata Gaul / Slang / Singkatan ke Kata Baku
SLANG_DICT = {
    # Negasi (Krusial untuk analisis sentimen)
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "kagak": "tidak",
    "g": "tidak",
    "gk": "tidak",
    "ngak": "tidak",
    "bkn": "bukan",
    "blm": "belum",
    "belom": "belum",
    "jgn": "jangan",

    # Intensifier / Penguat makna
    "bgt": "banget",
    "bgtt": "banget",
    "bgttt": "banget",
    "bangett": "banget",
    "syekali": "sekali",
    "sekaliii": "sekali",
    "bener": "benar",
    "beneran": "benar",
    "polll": "banget",
    "pol": "banget",

    # Kualitas Positif
    "bgs": "bagus",
    "baguuus": "bagus",
    "mantul": "mantap",
    "mantull": "mantap",
    "mantapp": "mantap",
    "mantappp": "mantap",
    "mantep": "mantap",
    "top": "bagus",
    "sip": "bagus",
    "sipp": "bagus",
    "sippp": "bagus",
    "oke": "bagus",
    "okeh": "bagus",
    "ok": "bagus",
    "gacor": "bagus",
    "rekomen": "rekomendasi",
    "recomended": "rekomendasi",
    "recommended": "rekomendasi",
    "recsel": "rekomendasi",
    "ori": "asli",
    "original": "asli",
    "makasih": "terima kasih",
    "thx": "terima kasih",
    "thanks": "terima kasih",

    # Kualitas Negatif
    "kecewaa": "kecewa",
    "kecewaaa": "kecewa",
    "kemcemwaa": "kecewa",
    "nyesel": "kecewa",
    "kapok": "kapok",
    "zonk": "kecewa",
    "ampas": "jelek",
    "gajelas": "tidak jelas",
    "ancur": "rusak",
    "cacat": "rusak",
    "cepet": "cepat",
    "cepettt": "cepat",
    "cpet": "cepat",
    "cpt": "cepat",
    "kilat": "cepat",
    "lemot": "lambat",
    "lemoot": "lambat",
    "lelet": "lambat",
    "lelettt": "lambat",

    # Entitas Produk & Aplikasi
    "brg": "barang",
    "brgnya": "barang",
    "brng": "barang",
    "item": "barang",
    "items": "barang",
    "product": "barang",
    "products": "barang",
    "produknya": "barang",
    "apk": "aplikasi",
    "app": "aplikasi",
    "apps": "aplikasi",
    "aplikasinya": "aplikasi",
    "olshop": "toko",
    "seller": "toko",
    "lapak": "toko",
    "cust": "pelanggan",
    "costumer": "pelanggan",
    "customer": "pelanggan",

    # Kata Hubung, Preposisi & Kata Ganti
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "tp": "tapi",
    "tpi": "tapi",
    "jd": "jadi",
    "jdi": "jadi",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "klo": "kalau",
    "kalo": "kalau",
    "dr": "dari",
    "pd": "pada",
    "sm": "sama",
    "sma": "sama",
    "sy": "saya",
    "aku": "saya",
    "gw": "saya",
    "gue": "saya",
    "km": "kamu",
    "lu": "kamu",
    "lo": "kamu",
    "u": "kamu",
    "org": "orang",
    "ad": "ada",
    "aja": "saja",
    "aj": "saja",
    "emg": "memang",
    "emang": "memang",
    "cm": "hanya",
    "cuma": "hanya",
    "cuman": "hanya",
    "trs": "terus",
    "lg": "lagi",
    "lgi": "lagi",
    "bisaa": "bisa",
    "bs": "bisa"
}

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Pipeline pembersihan teks ulasan:
    1. Case Folding (lowercase)
    2. Hapus URL, email, tag HTML
    3. De-lengthening huruf berulang (cth: 'bangetttt' -> 'banget', 'baguuusss' -> 'bagus')
    4. Hapus angka & tanda baca (hanya menyisakan alfabet a-z dan spasi)
    5. Tokenisasi kata
    6. Normalisasi slang / singkatan ke kata baku (cth: 'ga'/'gak' -> 'tidak', 'bgt' -> 'banget')
    7. Filter kata pendek (panjang > 1) & Stopwords Removal (opsional)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase (Case Folding)
    text = text.lower()
    
    # 2. Hapus URL, Email, Tag HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Normalisasi karakter berulang >= 3 kali (De-lengthening)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 4. Hapus angka, tanda baca, simbol khusus (sisakan a-z dan spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 5. Tokenisasi
    tokens = text.split()
    
    # 6. Normalisasi Slang & Konsonan/Vokal Ganda
    normalized_tokens = []
    for t in tokens:
        # Reduksi konsonan ganda di akhir kata jika ada (cth: 'bangett' -> 'banget', 'mantull' -> 'mantul')
        reduced = re.sub(r'([bcdfghjklmnpqrstvwxyz])\1+$', r'\1', t)
        # Reduksi vokal ganda di akhir kata jika panjang > 3 (cth: 'kecewaa' -> 'kecewa')
        if len(reduced) > 3:
            reduced = re.sub(r'([aeiou])\1+$', r'\1', reduced)
        
        normalized_word = SLANG_DICT.get(reduced, SLANG_DICT.get(t, reduced))
        normalized_tokens.append(normalized_word)
        
    # 7. Stopword removal & panjang kata > 1
    if remove_stopwords:
        tokens = [t for t in normalized_tokens if t not in INDONESIAN_STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in normalized_tokens if len(t) > 1]
        
    cleaned = " ".join(tokens)
    return cleaned
