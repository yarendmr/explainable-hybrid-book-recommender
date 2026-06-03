"""
Kitap öneri sistemi için demo veri üretici.
Gerçek Amazon Books verisi indirilmeden önce sistemi test etmek için
kitap türlerine dayalı sentetik ürün, kullanıcı ve etkileşim verisi üretir.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

random.seed(42)


BOOK_TEMPLATES = {
    "Fiction": [
        {
            "title_prefix": "Sessiz Sokaklar",
            "authors": ["Elif Karaca", "Murat Demir", "Aylin Yılmaz", "Kemal Arslan"],
            "desc_template": "{author} tarafından kaleme alınan bu roman, şehir hayatı, aile bağları ve kişisel dönüşüm temalarını sürükleyici bir anlatımla işler.",
            "style": "Roman / kurgu",
            "usage": "Günlük okuma ve edebi kurgu",
            "features": "Karakter odaklı anlatım, akıcı dil, duygusal derinlik",
            "price_range": (7, 22),
        },
        {
            "title_prefix": "Kayıp Defter",
            "authors": ["Selin Acar", "Barış Koç", "Nehir Şahin", "Deniz Aksoy"],
            "desc_template": "{author} imzalı bu eser, geçmişle yüzleşme ve insan ilişkileri üzerine kurulu çağdaş bir kurgu romanıdır.",
            "style": "Çağdaş kurgu",
            "usage": "Roman okumayı seven kullanıcılar",
            "features": "Duygusal anlatım, güçlü karakter gelişimi, gerçekçi atmosfer",
            "price_range": (8, 24),
        },
    ],
    "Science Fiction": [
        {
            "title_prefix": "Mars Günlüğü",
            "authors": ["Kerem Altan", "Ece Yıldız", "Can Ersoy", "Derya Kılıç"],
            "desc_template": "{author} tarafından yazılan bu bilim kurgu romanı, Mars kolonileri, yapay zeka ve insanlığın geleceği üzerine heyecanlı bir hikaye sunar.",
            "style": "Bilim kurgu",
            "usage": "Teknoloji ve uzay temalı okuma",
            "features": "Uzay kolonisi, yapay zeka, gelecek kurgusu",
            "price_range": (10, 30),
        },
        {
            "title_prefix": "Yapay Zekanın Gölgesi",
            "authors": ["Melis Kaya", "Arda Özkan", "Seda Yalçın", "Ozan Tekin"],
            "desc_template": "{author} imzalı bu kitap, insan-makine ilişkisini, etik kararları ve distopik gelecek senaryolarını konu alır.",
            "style": "Distopik bilim kurgu",
            "usage": "Yapay zeka ve gelecek temalı okuma",
            "features": "Distopya, etik ikilem, teknoloji eleştirisi",
            "price_range": (11, 32),
        },
    ],
    "Fantasy": [
        {
            "title_prefix": "Ejderha Kapısı",
            "authors": ["Berk Tuna", "İrem Güneş", "Alp Erden", "Nazlı Korkmaz"],
            "desc_template": "{author} tarafından yazılan bu fantastik roman, büyülü krallıklar, kadim sırlar ve destansı bir yolculuk etrafında şekillenir.",
            "style": "Fantastik",
            "usage": "Fantastik evren ve macera okuması",
            "features": "Büyü, ejderhalar, epik yolculuk",
            "price_range": (9, 28),
        },
        {
            "title_prefix": "Gümüş Krallık",
            "authors": ["Aslı Duran", "Mert Bilgin", "Yağmur Çelik", "Tolga Başar"],
            "desc_template": "{author} imzalı bu eser, krallık mücadeleleri, kahramanlık ve büyülü nesneler üzerine kurulu sürükleyici bir fantastik romandır.",
            "style": "Epik fantastik",
            "usage": "Seri kitap ve fantastik kurgu sevenler",
            "features": "Epik atmosfer, krallık çatışması, büyülü dünya",
            "price_range": (10, 31),
        },
    ],
    "Mystery & Thriller": [
        {
            "title_prefix": "Son İpucu",
            "authors": ["Serkan Işık", "Gizem Arı", "Emre Polat", "Mina Kaplan"],
            "desc_template": "{author} tarafından yazılan bu polisiye roman, karmaşık bir cinayet soruşturması ve beklenmedik olay örgüsüyle ilerler.",
            "style": "Polisiye / gerilim",
            "usage": "Gizem ve dedektiflik hikayesi",
            "features": "Dedektiflik, suç, sürükleyici final",
            "price_range": (8, 25),
        },
        {
            "title_prefix": "Karanlık Dosya",
            "authors": ["Cemre Doğan", "Onur Eren", "Pelin Savaş", "Hakan Yüce"],
            "desc_template": "{author} imzalı bu gerilim kitabı, gizli dosyalar, psikolojik baskı ve çözülmesi zor sırlarla ilerleyen tempolu bir anlatı sunar.",
            "style": "Psikolojik gerilim",
            "usage": "Heyecan ve gizem odaklı okuma",
            "features": "Gerilim, gizem, ters köşe anlatım",
            "price_range": (9, 27),
        },
    ],
    "Romance": [
        {
            "title_prefix": "Bir Yaz Akşamı",
            "authors": ["Zeynep Uçar", "Ela Bozkurt", "Eylül Sarı", "Merve Aydın"],
            "desc_template": "{author} tarafından kaleme alınan bu romantik roman, yaz tatili, ikinci şanslar ve duygusal ilişkiler üzerine sıcak bir hikaye anlatır.",
            "style": "Romantik roman",
            "usage": "Duygusal ve romantik okuma",
            "features": "Aşk hikayesi, sıcak atmosfer, kolay okuma",
            "price_range": (7, 21),
        },
        {
            "title_prefix": "Kalbimdeki Mektup",
            "authors": ["Burcu Kaan", "Ayşe Tan", "Sibel Ergin", "Dilan Öz"],
            "desc_template": "{author} imzalı bu kitap, geçmişten gelen bir mektup üzerinden aşk, özlem ve yeni başlangıçları konu alır.",
            "style": "Duygusal romantik kurgu",
            "usage": "Romantik ve hafif kurgu sevenler",
            "features": "Aşk, özlem, duygusal bağ",
            "price_range": (8, 23),
        },
    ],
    "Horror": [
        {
            "title_prefix": "Karanlık Ev",
            "authors": ["Tolga Kara", "Ekin Soylu", "İdil Taş", "Bora Deniz"],
            "desc_template": "{author} tarafından yazılan bu korku romanı, terk edilmiş bir ev, açıklanamayan olaylar ve giderek artan gerilim üzerine kuruludur.",
            "style": "Korku",
            "usage": "Korku ve karanlık atmosfer okuması",
            "features": "Tekinsiz atmosfer, doğaüstü olaylar, gerilim",
            "price_range": (8, 26),
        },
        {
            "title_prefix": "Gece Yarısı Günlüğü",
            "authors": ["Sarp Gök", "Ceren Yaman", "Oğuzhan Çınar", "Defne Kurt"],
            "desc_template": "{author} imzalı bu eser, kabuslar, gizemli günlükler ve psikolojik korku unsurlarıyla ilerler.",
            "style": "Psikolojik korku",
            "usage": "Karanlık kurgu ve gerilim sevenler",
            "features": "Psikolojik korku, gizem, karanlık tema",
            "price_range": (9, 28),
        },
    ],
    "History": [
        {
            "title_prefix": "İmparatorluğun İzleri",
            "authors": ["Ahmet Ergin", "Fatma Keskin", "Orhan Bulut", "Leyla Özdemir"],
            "desc_template": "{author} tarafından hazırlanan bu tarih kitabı, imparatorlukların yükselişi, toplumsal değişim ve siyasi dönüşümler üzerine kapsamlı bir anlatım sunar.",
            "style": "Tarih",
            "usage": "Tarih ve toplum okuması",
            "features": "Tarihsel analiz, dönem anlatımı, kaynak odaklı içerik",
            "price_range": (12, 35),
        },
        {
            "title_prefix": "Savaş ve Toplum",
            "authors": ["Mustafa Akın", "Nesrin Bal", "Ferhat Şen", "Gül Aras"],
            "desc_template": "{author} imzalı bu kitap, savaşların toplumlar üzerindeki etkisini, ekonomik değişimleri ve kültürel sonuçları inceler.",
            "style": "Tarihsel inceleme",
            "usage": "Akademik ve genel tarih okuması",
            "features": "Savaş tarihi, toplum analizi, dönemsel bakış",
            "price_range": (13, 38),
        },
    ],
    "Biography": [
        {
            "title_prefix": "Bir Hayatın Notları",
            "authors": ["Gökhan Efe", "Suna Kıraç", "İlker Baş", "Nilay Atıl"],
            "desc_template": "{author} tarafından yazılan bu biyografi, ilham verici bir yaşam öyküsünü, dönüm noktaları ve kişisel mücadeleler üzerinden anlatır.",
            "style": "Biyografi",
            "usage": "Gerçek yaşam hikayesi okuması",
            "features": "Yaşam öyküsü, ilham, kişisel mücadele",
            "price_range": (10, 30),
        },
        {
            "title_prefix": "Yolculuğum",
            "authors": ["Asuman Tek", "Rıza Solmaz", "Begüm Sezer", "Kaan Uslu"],
            "desc_template": "{author} imzalı bu otobiyografik eser, başarı, kayıp, öğrenme ve yeniden başlama temalarını kişisel bir dille aktarır.",
            "style": "Otobiyografi / anı",
            "usage": "Anı ve biyografi seven okuyucular",
            "features": "Anı, kişisel gelişim, gerçek deneyim",
            "price_range": (9, 29),
        },
    ],
    "Psychology": [
        {
            "title_prefix": "Zihnin Haritası",
            "authors": ["Dr. Deniz Akman", "Dr. Pınar Eren", "Dr. Bora Yalın", "Dr. Sema Acar"],
            "desc_template": "{author} tarafından hazırlanan bu psikoloji kitabı, davranış, duygu yönetimi ve insan zihninin işleyişini anlaşılır örneklerle açıklar.",
            "style": "Psikoloji",
            "usage": "Psikoloji ve davranış bilimi okuması",
            "features": "Duygu yönetimi, davranış analizi, anlaşılır örnekler",
            "price_range": (11, 34),
        },
        {
            "title_prefix": "Duyguların Dili",
            "authors": ["Dr. Elvan Kök", "Dr. Mert Ulaş", "Dr. Cansu Arı", "Dr. Eren Bal"],
            "desc_template": "{author} imzalı bu eser, stres, iletişim, öz farkındalık ve psikolojik dayanıklılık konularını ele alır.",
            "style": "Uygulamalı psikoloji",
            "usage": "Kişisel farkındalık ve psikoloji okuması",
            "features": "Stres yönetimi, iletişim, öz farkındalık",
            "price_range": (12, 36),
        },
    ],
    "Self-Help": [
        {
            "title_prefix": "Küçük Alışkanlıklar",
            "authors": ["Selçuk Tamer", "Ebru Kaya", "Yusuf Ekin", "Bade Arslan"],
            "desc_template": "{author} tarafından yazılan bu kişisel gelişim kitabı, alışkanlık kazanma, motivasyon ve sürdürülebilir başarı üzerine pratik öneriler sunar.",
            "style": "Kişisel gelişim",
            "usage": "Motivasyon ve alışkanlık geliştirme",
            "features": "Alışkanlık, üretkenlik, motivasyon",
            "price_range": (9, 27),
        },
        {
            "title_prefix": "Odaklanma Sanatı",
            "authors": ["Alev Yıldırım", "Mert Karahan", "Sıla Tekin", "Okan Güler"],
            "desc_template": "{author} imzalı bu eser, zaman yönetimi, odaklanma ve verimli çalışma becerilerini günlük yaşama uyarlanabilir şekilde anlatır.",
            "style": "Üretkenlik",
            "usage": "Verimli çalışma ve kişisel gelişim",
            "features": "Odaklanma, zaman yönetimi, hedef belirleme",
            "price_range": (10, 29),
        },
    ],
    "Business": [
        {
            "title_prefix": "Girişimcinin Yol Haritası",
            "authors": ["Burak Er", "Derya Ünal", "Selim Aydın", "Cansu Koç"],
            "desc_template": "{author} tarafından hazırlanan bu iş kitabı, girişimcilik, pazarlama ve büyüme stratejilerini örneklerle açıklar.",
            "style": "İş ve girişimcilik",
            "usage": "Girişimcilik ve iş dünyası okuması",
            "features": "Startup, pazarlama, iş modeli",
            "price_range": (12, 40),
        },
        {
            "title_prefix": "Stratejik Yönetim Notları",
            "authors": ["Mehmet Arı", "Gizem Uslu", "Koray Baş", "İpek Demir"],
            "desc_template": "{author} imzalı bu kitap, yönetim, liderlik ve karar alma süreçlerini iş dünyası örnekleriyle ele alır.",
            "style": "Yönetim",
            "usage": "İş, liderlik ve kariyer okuması",
            "features": "Liderlik, strateji, yönetim becerileri",
            "price_range": (14, 42),
        },
    ],
    "Children": [
        {
            "title_prefix": "Minik Kaşifler",
            "authors": ["Pelin Mutlu", "Ayhan Sevinç", "Güneş Kara", "Duru Yalın"],
            "desc_template": "{author} tarafından yazılan bu çocuk kitabı, merak, arkadaşlık ve keşfetme duygusunu sade bir dille anlatır.",
            "style": "Çocuk kitabı",
            "usage": "Çocuklara yönelik okuma",
            "features": "Eğlenceli dil, arkadaşlık, keşif",
            "price_range": (5, 18),
        },
        {
            "title_prefix": "Renkli Orman",
            "authors": ["Esra Kılıç", "Tuna Gök", "Meltem Arı", "Umut Şen"],
            "desc_template": "{author} imzalı bu resimli çocuk kitabı, doğa sevgisi, yardımlaşma ve hayal gücü temalarını işler.",
            "style": "Resimli çocuk kitabı",
            "usage": "Çocuklar için hikaye okuma",
            "features": "Doğa sevgisi, hayal gücü, renkli anlatım",
            "price_range": (6, 20),
        },
    ],
}


DEMO_USERS = [
    {"user_id": "U001", "display_name": "Ayşe D."},
    {"user_id": "U002", "display_name": "Yaren S."},
    {"user_id": "U003", "display_name": "Mehmet K."},
    {"user_id": "U004", "display_name": "Zeynep A."},
    {"user_id": "U005", "display_name": "Fatma Ö."},
    {"user_id": "U006", "display_name": "Ali R."},
    {"user_id": "U007", "display_name": "Elif T."},
    {"user_id": "U008", "display_name": "Can B."},
    {"user_id": "U009", "display_name": "Selin Y."},
    {"user_id": "U010", "display_name": "Hasan M."},
]


USER_PREFERENCES = {
    "U001": ["Science Fiction", "Fantasy", "Fiction"],
    "U002": ["Romance", "Fiction", "Self-Help"],
    "U003": ["Mystery & Thriller", "Horror", "Psychology"],
    "U004": ["Psychology", "Self-Help", "Biography"],
    "U005": ["History", "Biography", "Fiction"],
    "U006": ["Business", "Self-Help", "Psychology"],
    "U007": ["Fantasy", "Science Fiction", "Children"],
    "U008": ["Mystery & Thriller", "Fiction", "History"],
    "U009": ["Romance", "Children", "Fiction"],
    "U010": ["Business", "Biography", "History"],
}


def build_product_text(product: dict) -> str:
    """Öneri bileşenleri için birleşik kitap metni oluşturur."""
    fields = [
        "title",
        "description",
        "category",
        "brand",
        "author",
        "style",
        "usage",
        "features",
        "details",
        "main_category",
    ]

    return " | ".join(
        str(product.get(field, "")).strip()
        for field in fields
        if str(product.get(field, "")).strip()
    )


def generate_products(n_total: int = 200) -> pd.DataFrame:
    """Kitap türlerine dayalı sentetik ürün verisi üretir."""
    products = []
    product_id_counter = 1

    categories = list(BOOK_TEMPLATES.items())
    base_count = n_total // len(categories)
    remainder = n_total % len(categories)

    for category_index, (category, templates) in enumerate(categories):
        category_count = base_count + (1 if category_index < remainder else 0)

        for item_index in range(category_count):
            template = templates[item_index % len(templates)]
            author = random.choice(template["authors"])
            price = round(random.uniform(*template["price_range"]), 2)
            edition_no = item_index // len(templates) + 1

            product_id = f"P{product_id_counter:04d}"
            product_id_counter += 1

            title = f"{template['title_prefix']} {edition_no}"
            description = template["desc_template"].format(author=author)

            product = {
                "product_id": product_id,
                "title": title,
                "description": description,
                "category": category,
                "brand": author,
                "author": author,
                "price": str(price),
                "average_rating": round(random.uniform(3.6, 4.9), 1),
                "rating_number": random.randint(20, 600),
                "features": template["features"],
                "details": f"{category} türünde demo kitap kaydı",
                "main_category": "Books",
                "color": "",
                "material": "Kitap",
                "style": template["style"],
                "usage": template["usage"],
                "image_url": "",
            }

            product["product_text"] = build_product_text(product)
            products.append(product)

    return pd.DataFrame(products)


def generate_interactions(
    products_df: pd.DataFrame,
    n_interactions_per_user: int = 80,
) -> pd.DataFrame:
    """Kullanıcı tercihlerine göre ağırlıklı etkileşim verisi üretir."""
    interactions = []
    base_time = datetime(2024, 1, 1).timestamp()
    all_categories = list(BOOK_TEMPLATES.keys())

    for user in DEMO_USERS:
        user_id = user["user_id"]
        preferred_categories = USER_PREFERENCES.get(user_id, all_categories)

        other_categories = [
            category
            for category in all_categories
            if category not in preferred_categories
        ]

        preferred_products = products_df[
            products_df["category"].isin(preferred_categories)
        ]["product_id"].tolist()

        other_products = products_df[
            products_df["category"].isin(other_categories)
        ]["product_id"].tolist()

        preferred_count = int(n_interactions_per_user * 0.80)

        selected_preferred = random.sample(
            preferred_products,
            min(preferred_count, len(preferred_products)),
        )

        remaining_count = n_interactions_per_user - len(selected_preferred)

        selected_other = random.sample(
            other_products,
            min(remaining_count, len(other_products)),
        )

        selected_products = selected_preferred + selected_other
        random.shuffle(selected_products)

        for index, product_id in enumerate(selected_products):
            product_category = products_df.loc[
                products_df["product_id"] == product_id,
                "category",
            ].values[0]

            if product_category in preferred_categories:
                rating = random.choices([3, 4, 4, 5, 5], k=1)[0]
            else:
                rating = random.choices([1, 2, 3, 3, 4], k=1)[0]

            timestamp = base_time + index * 3600 + random.randint(0, 3600)

            interactions.append({
                "user_id": user_id,
                "product_id": product_id,
                "rating": rating,
                "timestamp": int(timestamp),
            })

    return pd.DataFrame(interactions)


def main() -> None:
    print("Kitap öneri sistemi için demo veri üretiliyor...")

    products_df = generate_products(n_total=200)
    print(f"  Ürün sayısı: {len(products_df)}")

    interactions_df = generate_interactions(products_df, n_interactions_per_user=80)
    print(f"  Etkileşim sayısı: {len(interactions_df)}")

    users_df = pd.DataFrame(DEMO_USERS)
    print(f"  Kullanıcı sayısı: {len(users_df)}")

    products_df.to_csv(PROCESSED / "products.csv", index=False, encoding="utf-8-sig")
    interactions_df.to_csv(PROCESSED / "interactions.csv", index=False, encoding="utf-8-sig")
    users_df.to_csv(PROCESSED / "users.csv", index=False, encoding="utf-8-sig")

    print("\nDosyalar kaydedildi:")
    print(f"  {PROCESSED / 'products.csv'}")
    print(f"  {PROCESSED / 'interactions.csv'}")
    print(f"  {PROCESSED / 'users.csv'}")
    print("\nSonraki adım: python scripts/build_embeddings.py")


if __name__ == "__main__":
    main()