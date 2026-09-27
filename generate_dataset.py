import os
import pandas as pd

def build_dataset():
    data = [
        # Product Reviews - Positive
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

        # Product Reviews - Negative
        ("Kualitas barang sangat buruk, bahan tipis dan gampang sobek. Kecewa sekali belanja di sini.", "negatif", "Produk - Pakaian"),
        ("Pengiriman sangat lambat, butuh waktu dua minggu baru sampai. Seller tidak kooperatif.", "negatif", "Produk - Pakaian"),
        ("Ukuran tidak sesuai deskripsi, terlalu kecil dan jahitan leher lepas. Minta refund dipersulit.", "negatif", "Produk - Pakaian"),
        ("Barang datang dalam keadaan rusak dan pecah. Packing cuma pakai plastik tipis tanpa bubble wrap.", "negatif", "Produk - Elektronik"),
        ("HP sering hang dan panas padahal baru dipakai sebentar. Tidak bisa dipakai buat main game.", "negatif", "Produk - Smartphone"),
        ("Headset suara cuma keluar sebelah, kualitas jelek dan mic tidak berfungsi dengan baik.", "negatif", "Produk - Aksesoris"),
        ("Sepatu solnya cepat lepas baru dipakai dua hari. Lem tidak kuat dan baunya menyengat.", "negatif", "Produk - Sepatu"),
        ("Baterai cepat drop dan tidak tahan lama. Barang tidak ori dan respon penjual sangat lambat.", "negatif", "Produk - Elektronik"),
        ("Charger tidak bisa digunakan, sama sekali tidak mengisi daya. Penjual di-chat tidak membalas.", "negatif", "Produk - Aksesoris"),
        ("Rasa makanan sudah basi dan baunya apek saat paket dibuka. Sangat mengecewakan dan merugikan.", "negatif", "Produk - Makanan/Minuman"),
        ("Tas tidak sesuai gambar promo, resleting gampang macet dan jahitan asal-asalan.", "negatif", "Produk - Tas"),
        ("Mouse sering disconnect sendiri dan kursor lompat-lompat. Produk cacat produksi kecewa bgt.", "negatif", "Produk - Komputer"),
        ("Powerbank kapasitas tidak sesuai keterangan, diisi sekali langsung habis. Jangan beli di sini!", "negatif", "Produk - Aksesoris"),
        ("Lotion pembersih buat kulit iritasi dan gatal-gatal. Produk diduga palsu tidak ada nomor BPOM.", "negatif", "Produk - Perawatan"),
        ("Keyboard tutsnya banyak yang keras dan tidak terdeteksi komputer. Minta retur ditolak toko.", "negatif", "Produk - Komputer"),
        ("Botol minum bocor saat diisi air. Plastik bau kimia tajam dan kualitasnya murahan.", "negatif", "Produk - Perabotan"),
        ("Helm ukurannya sempit sekali dan kacanya baret-baret saat dibuka dari dus. Kecewa parah.", "negatif", "Produk - Otomotif"),
        ("Lampu baru dipasang langsung mati dan mati total. Tidak worth it sama sekali.", "negatif", "Produk - Elektronik"),
        ("Parfum baunya menyengat seperti alkohol murni dan hilang dalam 5 menit. Tidak merekomendasikan.", "negatif", "Produk - Perawatan"),
        ("Kipas angin suaranya sangat bising dan mesin cepat panas. Pelayanan toko buruk sekali.", "negatif", "Produk - Elektronik"),

        # Application Reviews - Positive
        ("Aplikasi toko online ini sangat membantu, antarmuka simpel, navigasi mudah dan transaksi cepat!", "positif", "Aplikasi - E-Commerce"),
        ("Fitur pembayaran lengkap, banyak voucher diskon gratis ongkir dan promo menarik setiap hari.", "positif", "Aplikasi - E-Commerce"),
        ("Proses pengiriman makanan sangat cepat, lacak driver real-time akurat dan promo melimpah!", "positif", "Aplikasi - Transportasi/Food"),
        ("Aplikasi dompet digital sangat mudah digunakan, transfer antar bank gratis dan top up instan.", "positif", "Aplikasi - Keuangan"),
        ("Update aplikasi terbaru bikin tampilan semakin keren dan loading jauh lebih cepat tanpa lag.", "positif", "Aplikasi - Utility"),
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

        # Application Reviews - Negative
        ("Aplikasi sering crash dan keluar sendiri saat checkout pembayaran. Sangat mengganggu!", "negatif", "Aplikasi - E-Commerce"),
        ("Setelah update terbaru aplikasi jadi lemot dan sering force close. Tolong segera diperbaiki developer.", "negatif", "Aplikasi - Utility"),
        ("Saldo terpotong tapi transaksi gagal dan uang tidak kembali. Customer service tidak bisa dihubungi!", "negatif", "Aplikasi - Keuangan"),
        ("Terlalu banyak iklan pop-up yang menutupi layar dan bikin aplikasi sering hang.", "negatif", "Aplikasi - Informasi"),
        ("Proses verifikasi KTP gagaal terus padahal foto sudah sangat jelas. Aplikasi mengecewakan.", "negatif", "Aplikasi - Keuangan"),
        ("Lokasi GPS driver ngawur dan peta sering error tidak akurat saat pesan ojek online.", "negatif", "Aplikasi - Transportasi/Food"),
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

    # Generate additional synthetic variations to create a robust dataset of 200+ samples
    expanded_data = []
    
    positive_templates = [
        ("Produk {item} ini luar biasa bagus! {detail} Sangat puuasss dengan pembelian ini.", "positif"),
        ("Aplikasi {app} berfungsi lancar tanpa bug. {detail} Worth it banget dipake tiap hari.", "positif"),
        ("Seller ramah dan fast respon. {item} sampai dengan aman dan mulus. {detail}", "positif"),
        ("Sangat merekomendasikan {item} ini. Kualitas juara dan {detail}.", "positif"),
        ("Pengalaman menggunakan {app} sangat menyenangkan. UI intuitif dan {detail}.", "positif"),
        ("Barang {item} sesuai ekspektasi, pengiriman super cepat dan packing rapi. {detail}", "positif"),
        ("Fitur pada {app} lengkap dan sangat membantu pekerjaan harian. {detail}", "positif"),
        ("Harga {item} bersahabat tapi kualitas mewah. {detail} Top banget!", "positif"),
    ]

    negative_templates = [
        ("Jangan beli {item} di toko ini! {detail} Sangat kecewa dan merugikan pembeli.", "negatif"),
        ("Aplikasi {app} buruk sekali, sering error dan {detail}. Developer mohon diperbaiki!", "negatif"),
        ("Barang {item} rusak saat diterima. Penjual slow respon dan {detail}.", "negatif"),
        ("Kecewa berat beli {item}. Tidak sesuai gambar dan {detail}.", "negatif"),
        ("Update aplikasi {app} malah makin jelek. Fitur sering crash dan {detail}.", "negatif"),
        ("Kualitas {item} sangat murahan, gampang rusak dan {detail}.", "negatif"),
        ("Pelayanan toko buruk, pengiriman lama dan {detail} untuk {item}.", "negatif"),
        ("Aplikasi {app} mengecewakan, sistem sering down dan {detail}.", "negatif")
    ]

    items = ["HP Smartphone", "Laptop Core i7", "Headset Bluetooth", "Sepatu Lari", "Kemeja Pria", "Kamera Mirrorless", "Jam Tangan Digital", "Powerbank 20000mAh", "Tas Ransel", "Kipas Angin Portable", "Earphone Wireless", "Mouse Gaming", "Keyboard Mechanical", "Pembersih Wajah", "Botol Termos"]
    apps = ["Belanja Online", "Dompet Digital", "Ojek Online", "Streaming Musik", "Bank Mobile", "Game Multiplayer", "Investasi Saham", "Lacak Paket", "Presensi Pegawai", "Konsultasi Dokter"]

    pos_details = [
        "Jahitan sangat rapi dan kainnya sejuk dipakai.",
        "Proses pengiriman kilat cuma butuh 1 hari kerja.",
        "Respon customer service sangat baik dan informatif.",
        "Semua fitur berjalan sempurna tanpa ada kendala.",
        "Kemasan dilapisi karton dan bubble wrap berlapis-lapis.",
        "Baterai tahan lama hingga dua hari pemakaian normal.",
        "Sangat simpel digunakan dan tampilan visual estetik.",
        "Diskon besar dan banyak promo cashback menarik."
    ]

    neg_details = [
        "Bahan tipis gampang sobek dan ukurannya ngaco.",
        "Pengiriman terhambat hingga dua minggu tanpa kepastian.",
        "Sering force close saat membuka menu utama.",
        "Layanan pelanggan tidak responsif dan diabaikan.",
        "Kemasan penyok dan barang di dalamnya cacat fisik.",
        "Baterai boros sekali dan mesin cepat panas.",
        "Navigasi aplikasi membingungkan dan tombol sering macet.",
        "Saldo terpotong padahal pesanan dibatalkan sistem."
    ]

    # Combine explicit reviews
    for text, sentiment, kat in data:
        expanded_data.append({"ulasan": text, "sentimen": sentiment, "kategori": kat})

    # Synthetic additions for balance and rich training set
    idx = 0
    for i in range(75):
        item = items[i % len(items)]
        detail = pos_details[i % len(pos_details)]
        tmpl, sent = positive_templates[i % len(positive_templates)]
        text = tmpl.format(item=item, app=apps[i % len(apps)], detail=detail)
        expanded_data.append({"ulasan": text, "sentimen": sent, "kategori": f"Produk/Aplikasi ({item})"})

        item_neg = items[(i + 3) % len(items)]
        detail_neg = neg_details[i % len(neg_details)]
        tmpl_neg, sent_neg = negative_templates[i % len(negative_templates)]
        text_neg = tmpl_neg.format(item=item_neg, app=apps[i % len(apps)], detail=detail_neg)
        expanded_data.append({"ulasan": text_neg, "sentimen": sent_neg, "kategori": f"Produk/Aplikasi ({item_neg})"})

    df = pd.DataFrame(expanded_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset_ulasan.csv", index=False, encoding="utf-8")
    print(f"Dataset successfully created with {len(df)} rows!")
    print(df["sentimen"].value_counts())

if __name__ == "__main__":
    build_dataset()
