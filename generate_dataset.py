"""
generate_dataset.py
-------------------
Script generator dataset untuk tugas Analisis Sentimen (Teks Mining).
Menghasilkan:
1. data/raw_data.csv      : Dataset ulasan mentah dan berantakan (dirty text)
                            berisi slang, typo, kapitalisasi acak, URL, HTML,
                            tanda baca berlebih, emotikon, dan stopwords.
2. data/test_cases_25.csv : 25 kasus uji berlabel (positif & negatif seimbang)
                            sesuai ketentuan Milestone 1.
3. data/dataset_ulasan.csv: Dataset ulasan yang siap digunakan untuk training app.
"""

import os
import random
import re
import pandas as pd

# Pastikan reproduktifitas
random.seed(42)

# ==============================================================================
# 1. TEMPLATE KONTEN ULASAN DASAR (PRODUK & APLIKASI)
# ==============================================================================
SEED_REVIEWS = [
    # PRODUK - POSITIF
    ("Sepatu sangat nyaman dipakai untuk olahraga dan lari jarak jauh. Sol empuk dan jahitan rapi.", "positif", "Produk - Sepatu"),
    ("Kualitas bahan baju sangat bagus, adem dan halus. Ukuran pas sesuai dengan deskripsi toko.", "positif", "Produk - Pakaian"),
    ("Pengiriman sangat cepat, respon seller ramah dan pesanannya dibungkus bubble wrap tebal.", "positif", "Produk - Elektronik"),
    ("Laptop berfungsi dengan baik tanpa kendala. RAM besar dan baterai awet tahan seharian.", "positif", "Produk - Laptop"),
    ("Kamera HP sangat jernih, hasil foto malam hari memuaskan. Seller sangat recommended!", "positif", "Produk - Smartphone"),
    ("Headset suara bass mantap dan treble jernih. Pelayanan toko profesional dan pengiriman kilat.", "positif", "Produk - Aksesoris"),
    ("Bahan kain katun asli, tidak panas saat dipakai siang hari. Pasti order lagi di toko ini.", "positif", "Produk - Pakaian"),
    ("Jam tangan ori 100%, mulus tanpa cacat. Garansi resmi terjamin dan paking sangat aman.", "positif", "Produk - Jam Tangan"),
    ("Charger fast charging bekerja sempurna. HP terisi penuh dalam waktu 30 menit saja.", "positif", "Produk - Aksesoris"),
    ("Kopi rasanya nikmat, aroma harum dan tanggal kadaluarsa masih panjang. Sangat memuaskan.", "positif", "Produk - Makanan/Minuman"),
    ("Tas bahannya tebal, retsleting lancar dan muat banyak barang. Sangat recommended seller!", "positif", "Produk - Tas"),
    ("Mouse wireless respon cepat dan desain ergonomis, tangan tidak pegal dipakai seharian.", "positif", "Produk - Komputer"),
    ("Powerbank kapasitas murni, pengisian cepat dan barang sesuai dengan ekspektasi saya.", "positif", "Produk - Aksesoris"),
    ("Sabun pembersih muka bikin kulit segar dan lembab. Pengiriman cepat dan seller cepat balas chat.", "positif", "Produk - Perawatan"),
    ("Keyboard mekanik responsif, suara tutus enak didengar dan lampu RGB terang memukau.", "positif", "Produk - Komputer"),
    ("Botol minum bahan BPA free, tidak bocor dan warna sangat bagus. Anak saya sangat suka.", "positif", "Produk - Perabotan"),
    ("Helm pas di kepala, kaca bening dan busa dalam empuk sekali. Keamanan terjamin mantap!", "positif", "Produk - Otomotif"),
    ("Lampu LED terang hemat energi, pemasangan mudah dan harga sangat terjangkau.", "positif", "Produk - Elektronik"),
    ("Parfum aromanya harum tahan lama sepanjang hari. Paket datang dengan selamat tanpa bocor.", "positif", "Produk - Perawatan"),
    ("Kipas angin anginnya kencang tapi tidak bising. Kualitas barang bagus sepadan dengan harganya.", "positif", "Produk - Elektronik"),

    # PRODUK - NEGATIF
    ("Kualitas barang sangat buruk, bahan tipis dan gampang sobek. Kecewa sekali belanja di sini.", "negatif", "Produk - Pakaian"),
    ("Pengiriman sangat lambat, butuh waktu dua minggu baru sampai. Seller tidak kooperatif.", "negatif", "Produk - Pakaian"),
    ("Ukuran tidak sesuai deskripsi, terlalu kecil dan jahitan leher lepas. Minta refund dipersulit.", "negatif", "Produk - Pakaian"),
    ("Barang datang dalam keadaan rusak dan pecah. Packing cuma pakai plastik tipis tanpa bubble wrap.", "negatif", "Produk - Elektronik"),
    ("HP cepat panas padahal baru dipakai sebentar. Baterai boros dan suka mati mendadak.", "negatif", "Produk - Smartphone"),
    ("Suara headset sebelah kiri mati total, tidak bisa dipakai sama sekali. Toko tidak bertanggung jawab.", "negatif", "Produk - Aksesoris"),
    ("Warna baju luntur parah saat dicuci pertama kali, nodanya merusak pakaian lain. Kualitas abal-abal.", "negatif", "Produk - Pakaian"),
    ("Jam tangan mati, jarum detik copot di dalam. Sangat menyesal beli di toko ini, rugi uang.", "negatif", "Produk - Jam Tangan"),
    ("Charger tidak berfungsi sama sekali saat dicolokkan ke listrik. Barang kw murahan.", "negatif", "Produk - Aksesoris"),
    ("Makanan sudah bau tengik dan kemasannya bocor. Tidak layak dikonsumsi sama sekali.", "negatif", "Produk - Makanan/Minuman"),
    ("Retsleting tas macet dan jahitan tali pundak lepas saat diisi buku. Jahitan sangat rapuh.", "negatif", "Produk - Tas"),
    ("Mouse sering tidak terdeteksi oleh laptop, kursor sering ngelag dan loncat-loncat sendiri.", "negatif", "Produk - Komputer"),
    ("Powerbank tidak bisa mengisi daya dan badannya menggembung berbahaya. Jangan beli di sini!", "negatif", "Produk - Aksesoris"),
    ("Kulit wajah jadi gatal dan memerah setelah memakai produk ini. Tidak cocok dan bikin iritasi parah.", "negatif", "Produk - Perawatan"),
    ("Keyboard tombol spasi macet tidak bisa ditekan, bahan plastik sangat murahan dan ringkih.", "negatif", "Produk - Komputer"),
    ("Botol minum bau plastik menyengat dan tutupnya bocor saat dimiringkan. Kualitas jelek.", "negatif", "Produk - Perabotan"),
    ("Kaca helm buram dan banyak baretan dari pabrik. Busa helm tipis dan tidak nyaman dipakai.", "negatif", "Produk - Otomotif"),
    ("Lampu baru dinyalakan lima menit langsung redup dan mati total. Kualitas sangat mengecewakan.", "negatif", "Produk - Elektronik"),
    ("Parfum wanginya menyengat bikin pusing, botolnya rembes saat pengiriman dan basah semua.", "negatif", "Produk - Perawatan"),
    ("Kipas angin suaranya berisik seperti helikopter dan baling-balingnya oblak. Mengecewakan.", "negatif", "Produk - Elektronik"),

    # APLIKASI - POSITIF
    ("Aplikasi e-commerce ini tampilannya sangat simpel, pembayaran cepat dan promo gratis ongkir banyak.", "positif", "Aplikasi - E-Commerce"),
    ("Aplikasi dompet digital sangat memudahkan transaksi harian, transfer antar bank gratis dan cepat.", "positif", "Aplikasi - Keuangan"),
    ("Aplikasi ojek online sangat responsif, navigasi peta akurat dan tarif perjalanan terjangkau.", "positif", "Aplikasi - Transportasi"),
    ("UI aplikasi sangat bersih, navigasi mudah dipahami bahkan untuk pemula. Kinerjanya stabil dan ringan.", "positif", "Aplikasi - Utility"),
    ("Customer service responsif membantu kendala akun saya dengan cepat dan ramah sekali.", "positif", "Aplikasi - Layanan"),
    ("Fitur pencarian barang akurat, rekomendasi produk sesuai selera dan checkout sangat praktis.", "positif", "Aplikasi - E-Commerce"),
    ("Sistem keamanan aplikasi sangat baik dengan verifikasi biometrik dan OTP yang responsif.", "positif", "Aplikasi - Keuangan"),
    ("Aplikasi edukasi ini sangat bermanfaat untuk belajar mandiri, materi lengkap dan video jelas.", "positif", "Aplikasi - Edukasi"),
    ("Game sangat seru, grafik memukau dan fitur multiplayer lancar tanpa masalah server.", "positif", "Aplikasi - Game"),
    ("Aplikasi streaming musik liriknya sinkron, audio jernih dan rekomendasi lagu sangat bagus.", "positif", "Aplikasi - Hiburan"),
    ("Aplikasi presensi kantor akurat dengan fitur GPS lokasi yang tepat dan rekap jam kerja mudah.", "positif", "Aplikasi - Produktivitas"),
    ("Aplikasi berita cepat update, artikel informatif dan tidak terlalu banyak iklan mengganggu.", "positif", "Aplikasi - Informasi"),
    ("Fitur lacak paket sangat membantu mengetahui lokasi kurir secara akurat dan tepat waktu.", "positif", "Aplikasi - Logistik"),
    ("Aplikasi kesehatan ini mempermudah konsultasi dengan dokter spesialis secara online dan cepat.", "positif", "Aplikasi - Kesehatan"),

    # APLIKASI - NEGATIF
    ("Aplikasi sering crash dan keluar sendiri saat checkout pembayaran. Sangat mengganggu!", "negatif", "Aplikasi - E-Commerce"),
    ("Setelah update terbaru aplikasi jadi lemot dan sering force close. Tolong segera diperbaiki developer.", "negatif", "Aplikasi - Utility"),
    ("Saldo terpotong tapi transaksi gagal dan uang tidak kembali. Customer service tidak bisa dihubungi!", "negatif", "Aplikasi - Keuangan"),
    ("Terlalu banyak iklan pop-up yang menutupi layar dan bikin aplikasi sering hang.", "negatif", "Aplikasi - Informasi"),
    ("Proses verifikasi KTP gagal terus padahal foto sudah sangat jelas. Aplikasi mengecewakan.", "negatif", "Aplikasi - Keuangan"),
    ("Lokasi GPS driver ngawur dan peta sering error tidak akurat saat pesan ojek online.", "negatif", "Aplikasi - Transportasi"),
    ("Aplikasi boros kuota dan memakan penyimpanan internal terlalu besar. Bikin HP cepat panas.", "negatif", "Aplikasi - Utility"),
    ("Voucher diskon tidak bisa digunakan padahal syarat dan ketentuan sudah terpenuhi. Promo hoax!", "negatif", "Aplikasi - E-Commerce"),
    ("Notifikasi spam terlalu banyak masuk tiap jam sangat mengganggu kenyamanan pengguna.", "negatif", "Aplikasi - E-Commerce"),
    ("Server sering down maintenance saat jam sibuk. Tidak bisa dipakai buat transfer uang penting.", "negatif", "Aplikasi - Keuangan"),
    ("Login sering gagal padahal password dan email sudah benar. Fitur lupa password juga error.", "negatif", "Aplikasi - Utility"),
    ("Fitur pencarian tidak akurat, barang yang muncul tidak sesuai dengan kata kunci yang diketik.", "negatif", "Aplikasi - E-Commerce"),
    ("Customer service robot tidak menyelesaikan masalah, hanya berputar-putar tanpa solusi.", "negatif", "Aplikasi - Layanan"),
    ("Aplikasi game sering disconnected dari server saat match padahal koneksi internet lancar.", "negatif", "Aplikasi - Game"),
    ("Riwayat transaksi sering hilang dan tidak dapat diunduh untuk laporan keuangan.", "negatif", "Aplikasi - Keuangan")
]

# ==============================================================================
# 2. NOISE INJECTION ENGINE (UNTUK MEMBUAT DATA MENTAH & KOTOR)
# ==============================================================================
SLANG_MAP = {
    "sangat": ["bgt", "bangett", "bgttt", "syekali", "bener-bener", "bgtt", "sumpahh", "polll"],
    "bagus": ["baguuus", "bgs", "mantull", "mantappp", "top markotop", "oke", "okeh","sippp", "mantep", "tep mantep", "gacor"],
    "tidak": ["ga", "gak", "nggak", "tdk", "kagak", "g", "gk"],
    "kecewa": ["kecewa bgt", "zonk", "kecewaaa", "nyesel", "kapok", "kemcemwaa", "kecewaa berat", "ampas", "gajelas"],
    "cepat": ["cepet", "cpet", "kilat", "cepettt", "cpt"],
    "lambat": ["lemot", "lelet", "lama bgt", "lelettt"],
    "rusak": ["ancur", "rusak parah", "cacat", "pecah", "error mulu", "gepeng", "penyok", "lecet", "baret"],
    "rekomendasi": ["rekomen", "recomended", "recsel", "recomended seller"],
    "toko": ["olshop", "seller", "lapak", "toko"],
    "barang": ["brg", "item", "produknya", "brgnya", "brng", "items", "product", "products"],
    "aplikasi": ["apk", "apps", "app-nya", "aplikasinya", "app", "aplikasi nya", "mobile app"]
}

ANGRY_PROFANITY_WORDS = [
    "sialan", "tai", "penipu", "tolol", "brengsek", "kampret", "bego",
]

URL_SAMPLES = [
    "https://tokopedia.link/xyz123",
    "http://shopee.co.id/item/99812",
    "https://youtu.be/review_unboxing",
    "www.bukalapak.com/p/fashion",
    "http://bit.ly/test-promo-ongkir",
    "https://www.blibli.com/_search?searchTerm=flazz",
    "https://www.blibli.com/p/sedap-wangi-beras-premium-10-kg",
    "https://www.blibli.com/p/rose-brand-minyak-goreng-pouch-2-l"
]

EMAIL_SAMPLES = [
    "cs-bantuan@tokopedia.co.id",
    "support@belanjaonline.com",
    "helpdesk@bankmobile.id",
    "admin.seller@gmail.com",
    "support@toko.co.id",
    "customer.care@blibli.com"
]

HTML_TAGS = [
    "<br>",
    "<br/>",
    "<b>MANTAP</b>",
    "<p>",
    "</p>",
    "<i>rekomended</i>",
    "<span>100% ori</span>",
    "<div>kecewa</div>"
]

POS_EMOJIS = ["👍", "🔥", "😍", "✨", "⭐⭐⭐⭐⭐", "💯", "🥰", ":)", ":D", "^_^", "🥳", "🤭", "✅", "😁", "😆"]
NEG_EMOJIS = ["😡", "👎", "💔", "⭐", "🤮", "😤", ":(", ":'(", "-_-", "🤬", "🖕", "💩", "):", ")':"]

def inject_dirty_noise(text: str, sentiment: str) -> str:
    """
    Mengubah teks bersih menjadi teks mentah (dirty text) yang realistis.
    Memasukkan typo, slang, emotikon, URL, email, tag HTML, kata umpatan, dan tanda baca berlebih.
    """
    dirty = text
    
    # 1. Substitusi kata formal dengan bahasa gaul / slang
    for formal, slangs in SLANG_MAP.items():
        if re.search(r'\b' + formal + r'\b', dirty, flags=re.IGNORECASE):
            if random.random() < 0.6:
                chosen_slang = random.choice(slangs)
                dirty = re.sub(r'\b' + formal + r'\b', chosen_slang, dirty, count=1, flags=re.IGNORECASE)
                
    # 2. Huruf berulang (lengthening) pada beberapa kata
    words = dirty.split()
    if words and random.random() < 0.5:
        target_idx = random.randint(0, len(words) - 1)
        word = words[target_idx]
        if len(word) > 3 and word.isalpha():
            # gandakan huruf vokal terakhir
            last_char = word[-1]
            words[target_idx] = word + (last_char * random.randint(2, 4))
            dirty = " ".join(words)
            
    # 3. Tanda baca berlebih (excessive punctuation)
    punct_list = ["!!!", "???", "...?!", "????", "!!!!", "...."]
    if random.random() < 0.5:
        dirty = dirty.rstrip(".!?") + " " + random.choice(punct_list)
        
    # 4. Sisipkan URL / Link unboxing / toko (15% peluang)
    if random.random() < 0.15:
        url = random.choice(URL_SAMPLES)
        dirty += f" (cek di {url})"
        
    # 5. Sisipkan Email CS / Pengaduan (10% peluang)
    if random.random() < 0.10:
        email = random.choice(EMAIL_SAMPLES)
        dirty += f" hubungi {email}"
        
    # 6. Sisipkan Tag HTML (12% peluang)
    if random.random() < 0.12:
        tag = random.choice(HTML_TAGS)
        dirty = f"{dirty} {tag}"
        
    # 7. Tambahkan Emotikon / Simbol Bintang sesuai sentimen
    if sentiment == "positif":
        emojis = " ".join(random.sample(POS_EMOJIS, k=random.randint(1, 3)))
    else:
        emojis = " ".join(random.sample(NEG_EMOJIS, k=random.randint(1, 3)))
    dirty = f"{dirty} {emojis}"

    # 7b. Sisipkan kata umpatan/kekesalan untuk ulasan negatif (25% peluang)
    if sentiment == "negatif" and random.random() < 0.25:
        curse = random.choice(ANGRY_PROFANITY_WORDS)
        if random.random() < 0.5:
            dirty = f"{curse}! {dirty}"
        else:
            dirty = f"{dirty} dasar {curse}!"
    
    # 8. Variasi Kapitalisasi (ALL CAPS, Title Case, atau acak)
    cap_rand = random.random()
    if cap_rand < 0.15:
        dirty = dirty.upper()  # Capslock emosi
    elif cap_rand < 0.25:
        # Acak huruf besar/kecil di awal kalimat
        dirty = dirty[0].lower() + dirty[1:] if len(dirty) > 1 else dirty
        
    # 9. Spasi ganda atau spasi di ujung
    if random.random() < 0.3:
        dirty = "  " + dirty + "   "
        
    return dirty

# ==============================================================================
# 3. EKSPANSI SINTETIK UNTUK DATASET LENGKAP (~550+ BARIS)
# ==============================================================================
def build_expanded_dataset(target_per_class: int = 300):
    """
    Membangun dataset ulasan dasar independen sebanyak target_per_class * 2
    (default: 300 positif + 300 negatif = 600 data ulasan).
    """
    pos_data = []
    neg_data = []

    # 1. Masukkan ulasan bibit awal terpisah per sentimen
    for text, sentiment, kat in SEED_REVIEWS:
        if sentiment == "positif":
            pos_data.append({"ulasan": text, "sentimen": sentiment, "kategori": kat})
        else:
            neg_data.append({"ulasan": text, "sentimen": sentiment, "kategori": kat})

    items = [
        "HP Smartphone", "Laptop Core i7", "Headset Bluetooth", "Sepatu Lari", 
        "Kemeja Pria", "Kamera Mirrorless", "Jam Tangan Digital", "Powerbank 20000mAh", 
        "Tas Ransel", "Kipas Angin Portable", "Earphone Wireless", "Mouse Gaming", 
        "Keyboard Mechanical", "Pembersih Wajah", "Botol Termos", "Smart TV 4K",
        "Tablet Android", "Blender Portable", "Jaket Hoodie", "Speaker Bluetooth"
    ]
    apps = [
        "Belanja Online", "Dompet Digital", "Ojek Online", "Streaming Musik", 
        "Bank Mobile", "Game Multiplayer", "Investasi Saham", "Lacak Paket", 
        "Presensi Pegawai", "Konsultasi Dokter", "Streaming Film", "Pemesanan Hotel"
    ]

    positive_templates = [
        ("Produk {item} ini luar biasa bagus! {detail} Sangat puas dengan pembelian ini.", "positif", "Produk"),
        ("Aplikasi {app} berfungsi lancar tanpa bug. {detail} Sangat membantu untuk kegiatan harian.", "positif", "Aplikasi"),
        ("Seller ramah dan respon kilat. {item} sampai dengan aman dan mulus. {detail}", "positif", "Produk"),
        ("Sangat merekomendasikan {item} ini. Kualitas juara dan {detail}.", "positif", "Produk"),
        ("Pengalaman menggunakan aplikasi {app} sangat menyenangkan. UI intuitif dan {detail}.", "positif", "Aplikasi"),
        ("Barang {item} sesuai ekspektasi, pengiriman super cepat dan packing rapi. {detail}", "positif", "Produk"),
        ("Fitur pada aplikasi {app} lengkap dan sangat mempermudah pekerjaan. {detail}", "positif", "Aplikasi"),
        ("Harga {item} bersahabat tapi kualitas mewah sekelas barang mahal. {detail}", "positif", "Produk"),
        ("Layanan top markotop, {item} berfungsi optimal tanpa kendala. {detail}", "positif", "Produk"),
        ("Desain aplikasi {app} minimalis, hemat memori dan {detail}. Wajib install!", "positif", "Aplikasi")
    ]

    negative_templates = [
        ("Jangan beli {item} di toko ini! {detail} Sangat kecewa dan merugikan pembeli.", "negatif", "Produk"),
        ("Aplikasi {app} buruk sekali, sering error dan {detail}. Tolong perbaiki segera!", "negatif", "Aplikasi"),
        ("Barang {item} rusak saat diterima. Penjual lambat merespon keluhan dan {detail}.", "negatif", "Produk"),
        ("Kecewa berat belanja {item}. Tidak sesuai foto dan deskripsi toko, {detail}.", "negatif", "Produk"),
        ("Update aplikasi {app} malah makin jelek. Fitur sering macet dan {detail}.", "negatif", "Aplikasi"),
        ("Kualitas {item} sangat murahan, gampang hancur dan {detail}. Menyesal beli.", "negatif", "Produk"),
        ("Pelayanan toko buruk, pengiriman lambat dan {detail} untuk produk {item}.", "negatif", "Produk"),
        ("Aplikasi {app} mengecewakan, sistem sering down mendadak dan {detail}.", "negatif", "Aplikasi"),
        ("Minta retur {item} berbelit-belit padahal {detail}. Sangat tidak profesional.", "negatif", "Produk"),
        ("Aplikasi {app} memakan baterai parah, boros kuota dan {detail}. Uninstalled!", "negatif", "Aplikasi")
    ]

    pos_details = [
        "Jahitan sangat rapi, material premium dan tahan lama.",
        "Proses pengiriman kilat cuma butuh 1 hari kerja langsung sampai.",
        "Respon customer service sangat baik, sopan dan solutif.",
        "Semua fitur berjalan sempurna tanpa ada kendala sama sekali.",
        "Kemasan dilapisi karton tebal dan bubble wrap berlapis-lapis.",
        "Baterai awet tahan seharian dan performanya sangat stabil.",
        "Sangat simpel digunakan dan tampilan visualnya sangat elegan.",
        "Banyak promo diskon besar dan cashback yang sangat menguntungkan.",
        "Suara jernih tidak ada distorsi dan koneksi sinyal stabil.",
        "Bahan sejuk di kulit dan tidak menimbulkan iritasi sama sekali."
    ]

    neg_details = [
        "Bahan tipis mudah robek dan ukurannya sama sekali tidak sesuai.",
        "Pengiriman terhambat hingga dua minggu tanpa kabar jelas dari kurir.",
        "Sering force close tiba-tiba saat membuka halaman utama.",
        "Layanan pelanggan lambat merespon dan komplain diabaikan begitu saja.",
        "Kemasan penyok dan barang di dalamnya cacat fisik saat dibuka.",
        "Baterai sangat boros dan suhu perangkat cepat panas membakar.",
        "Navigasi aplikasi rumit, lambat dan tombol sering tidak merespon.",
        "Saldo terpotong padahal pesanan dibatalkan sepihak oleh sistem.",
        "Baut kendor dan beberapa komponen lepas dari tempatnya.",
        "Warna tidak sesuai pesanan dan kualitas jahitan sangat kasar."
    ]

    # 2. Tambahkan variasi ulasan positif hingga mencapai target_per_class
    idx_pos = 0
    while len(pos_data) < target_per_class:
        item = items[idx_pos % len(items)]
        app = apps[idx_pos % len(apps)]
        det = pos_details[idx_pos % len(pos_details)]
        tmpl, sent, tipe = positive_templates[idx_pos % len(positive_templates)]
        text = tmpl.format(item=item, app=app, detail=det)
        kat = f"{tipe} - {item if tipe == 'Produk' else app}"
        pos_data.append({"ulasan": text, "sentimen": sent, "kategori": kat})
        idx_pos += 1

    # 3. Tambahkan variasi ulasan negatif hingga mencapai target_per_class
    idx_neg = 0
    while len(neg_data) < target_per_class:
        item = items[(idx_neg + 5) % len(items)]
        app = apps[(idx_neg + 3) % len(apps)]
        det = neg_details[idx_neg % len(neg_details)]
        tmpl, sent, tipe = negative_templates[idx_neg % len(negative_templates)]
        text = tmpl.format(item=item, app=app, detail=det)
        kat = f"{tipe} - {item if tipe == 'Produk' else app}"
        neg_data.append({"ulasan": text, "sentimen": sent, "kategori": kat})
        idx_neg += 1

    all_data = pos_data + neg_data
    random.shuffle(all_data)
    return pd.DataFrame(all_data)

# ==============================================================================
# 4. FUNGSI UTAMA GENERASI FILE
# ==============================================================================
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("=" * 60)
    print(" [*] MEMULAI GENERASI DATASET MENTAH (RAW DATA > 500 & TEST CASES)")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # 1. Bangun 600 ulasan dasar secara mandiri (300 Positif, 300 Negatif)
    # Catatan: Sama sekali TIDAK membaca atau mengubah data/dataset_ulasan.csv!
    print("[+] Membuat 600 ulasan dasar secara mandiri (300 Positif, 300 Negatif)...")
    df_base = build_expanded_dataset(target_per_class=300)

    print(f"    Total data yang digenerate: {len(df_base)} ulasan")
    print(f"    Distribusi Sentimen:\n{df_base['sentimen'].value_counts().to_string()}\n")

    # 2. Bangun data/raw_data.csv (Data Mentah & Kotor / Dirty Text > 500 data)
    print("[*] Menginjeksikan noise teks mentah (slang, typo, URL, HTML, umpatan, emotikon)...")
    raw_records = []
    for idx, row in df_base.iterrows():
        dirty_text = inject_dirty_noise(str(row["ulasan"]), str(row["sentimen"]))
        raw_records.append({
            "id": idx + 1,
            "ulasan_mentah": dirty_text,
            "sentimen": row["sentimen"],
            "kategori": row.get("kategori", "Umum")
        })

    df_raw = pd.DataFrame(raw_records)
    raw_path = "data/raw_data.csv"
    df_raw.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"[OK] Berhasil membuat {raw_path} ({len(df_raw)} baris data mentah > 500)")

    # 3. Bangun data/test_cases_25.csv (25 Kasus Uji Berlabel untuk Milestone 1)
    print("\n[*] Menyiapkan 25 Kasus Uji Berlabel (Test Cases) untuk Milestone 1...")
    pos_samples = df_raw[df_raw["sentimen"] == "positif"].sample(n=13, random_state=42)
    neg_samples = df_raw[df_raw["sentimen"] == "negatif"].sample(n=12, random_state=42)
    
    df_test_cases = pd.concat([pos_samples, neg_samples]).sample(frac=1.0, random_state=123).reset_index(drop=True)
    df_test_cases["test_case_id"] = [f"TC_{i+1:02d}" for i in range(len(df_test_cases))]
    
    cols = ["test_case_id", "ulasan_mentah", "sentimen", "kategori"]
    df_test_cases = df_test_cases[cols]
    
    test_cases_path = "data/test_cases_25.csv"
    df_test_cases.to_csv(test_cases_path, index=False, encoding="utf-8")
    print(f"[OK] Berhasil membuat {test_cases_path} ({len(df_test_cases)} kasus uji)")

    # Tampilkan preview 3 sampel dirty data
    print("\n[*] CONTOH HASIL TEKS MENTAH (DIRTY TEXT SAMPLE):")
    print("-" * 60)
    for i in range(3):
        sample = df_test_cases.iloc[i]
        safe_preview = sample['ulasan_mentah'].encode('ascii', errors='replace').decode('ascii')
        print(f"[{sample['test_case_id']}] Sentimen: {sample['sentimen'].upper()} ({sample['kategori']})")
        print(f"Teks: {safe_preview}\n")

    print("=" * 60)
    print("[SUCCESS] GENERASI BERHASIL:")
    print("   1. data/raw_data.csv       -> 600 data mentah kotor (> 500 baris)")
    print("   2. data/test_cases_25.csv  -> 25 kasus uji untuk Milestone 1")
    print("   *  data/dataset_ulasan.csv -> DIBIARKAN UTUH FRESH DARI GIT (TIDAK DISENTUH)")
    print("=" * 60)

if __name__ == "__main__":
    main()
