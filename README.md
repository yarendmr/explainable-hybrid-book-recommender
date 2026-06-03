
# Amazon Reviews 2023 Books Veri Seti Üzerinde Açıklanabilir Hibrit Kitap Öneri Sistemi

Bu proje, Amazon Reviews 2023 Books veri seti üzerinde çalışan açıklanabilir hibrit bir kitap öneri sistemi uygulamasıdır. Sistem; kullanıcı davranışları, kitap metadata bilgileri, anlamsal benzerlik, popülerlik ve tür/niyet eşleşmesi sinyallerini birleştirerek kitap önerileri üretir. Önerilen kitapların neden önerildiği, kısa gerekçe ve isteğe bağlı LLM destekli ayrıntılı açıklama ile kullanıcıya sunulur.

Sistem doğrudan büyük dil modeliyle kitap üretmez. Öneri listesi hibrit öneri motoru tarafından gerçek katalog kitapları arasından oluşturulur. Qwen2.5-0.5B-Instruct modeli yalnızca kullanıcı ayrıntılı açıklama istediğinde, seçilen kitap için öneri gerekçesini doğal Türkçe ile açıklamak amacıyla kullanılır.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Kullanılan Veri Seti](#kullanılan-veri-seti)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Öneri Bileşenleri](#öneri-bileşenleri)
- [Hibrit Skor Formülleri](#hibrit-skor-formülleri)
- [Açıklama Üretimi](#açıklama-üretimi)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Veri Hazırlama](#veri-hazırlama)
- [Embedding Oluşturma](#embedding-oluşturma)
- [Uygulamayı Çalıştırma](#uygulamayı-çalıştırma)
- [Kullanım Rehberi](#kullanım-rehberi)
- [API Uç Noktaları](#api-uç-noktaları)
- [Değerlendirme](#değerlendirme)
- [Testler](#testler)
- [Çevre Değişkenleri](#çevre-değişkenleri)
- [Sınırlılıklar](#sınırlılıklar)
- [Gelecek Çalışmalar](#gelecek-çalışmalar)

---

## Özellikler

- Amazon Reviews 2023 Books veri seti ile çalışma
- Kullanıcı geçmişine dayalı kişiselleştirilmiş kitap önerisi
- Yeni kullanıcılar için öneri
- Türkçe doğal dil sorguları ile kitap arama
- Benzer kitap önerileri
- Kitap karşılaştırma tablosu
- Item-based collaborative filtering
- TF-IDF tabanlı content-based filtering
- Çok dilli SentenceTransformer tabanlı semantic retrieval
- Popülerlik ve tür/niyet eşleşmesi sinyalleri
- Baskın öneri sinyaline dayalı kısa gerekçe
- Qwen2.5-0.5B-Instruct ile isteğe bağlı ayrıntılı açıklama
- FastAPI backend
- Streamlit frontend
- Top-K değerlendirme metrikleri
- Pytest testleri

---

## Kullanılan Veri Seti

Projede `McAuley-Lab/Amazon-Reviews-2023` veri setinin `Books` alt kategorisi kullanılmaktadır.

Kullanılan temel veri kaynakları:

```text
raw_review_Books
raw_meta_Books
````

Review verisinden kullanılan temel alanlar:

* `user_id`
* `parent_asin`
* `rating`
* `timestamp`

Metadata verisinden kullanılan temel alanlar:

* `title`
* `description`
* `category`
* `author/store`
* `price`
* `average_rating`
* `rating_number`
* `image`

Books kategorisi büyük olduğu için veri setinin tamamı belleğe alınmaz. `scripts/download_amazon_subset.py` betiği, Hugging Face Datasets streaming yaklaşımı ile sınırlı sayıda review ve metadata kaydı tarar. Daha sonra kullanıcı–kitap etkileşimleri ve kitap metadata kayıtları eşleştirilerek öneri sistemine uygun alt veri kümesi oluşturulur.

Varsayılan veri hazırlama yaklaşımı:

* En fazla 300.000 review kaydı taranır.
* En fazla 1.200.000 metadata kaydı taranır.
* En az 3 etkileşime sahip olmayan kullanıcılar çıkarılır.
* En az 3 değerlendirme almamış kitaplar çıkarılır.
* En fazla 1.000 kitap ve 10.000 etkileşim korunur.

Bu değerler komut satırı argümanları ile değiştirilebilir.

---

## Sistem Mimarisi

Sistem dört temel katmandan oluşur:

### 1. Veri Katmanı

İşlenmiş veri ve embedding dosyaları bu katmanda tutulur.

```text
data/processed/products.csv
data/processed/interactions.csv
data/processed/users.csv
data/embeddings/product_embeddings.npy
data/embeddings/product_ids.txt
```

### 2. Öneri Motoru Katmanı

Aşağıdaki sinyaller ayrı ayrı hesaplanır ve hibrit skor altında birleştirilir:

* Collaborative Filtering
* Content-Based Filtering
* Semantic Retrieval
* Popularity Scoring
* Category / Intent Matching

### 3. Açıklama Katmanı

Sistem iki seviyeli açıklama üretir:

* Kısa gerekçe: LLM kullanılmadan, baskın öneri sinyaline göre üretilir.
* Ayrıntılı açıklama: Kullanıcı talep ettiğinde Qwen2.5-0.5B-Instruct ile üretilir.

### 4. Servis ve Arayüz Katmanı

* Backend: FastAPI
* Frontend: Streamlit

Genel çalışma akışı:

```text
Amazon Reviews 2023 Books
        ↓
Veri ön işleme
        ↓
Kitap metadata temsili + kullanıcı etkileşimleri
        ↓
CF + CBF + SEM + POP + CAT
        ↓
Hibrit skor hesaplama
        ↓
Öneri listesi
        ↓
Kısa gerekçe / isteğe bağlı LLM açıklaması
        ↓
FastAPI + Streamlit arayüz
```

---

## Öneri Bileşenleri

### Collaborative Filtering

İşbirlikçi filtreleme bileşeni item-based collaborative filtering yaklaşımına dayanır.

* Kullanıcı–kitap pivot matrisi oluşturulur.
* Rating değerleri davranışsal ağırlığa dönüştürülür.
* Kitap–kitap kosinüs benzerliği hesaplanır.
* Kullanıcının geçmiş etkileşimleri üzerinden aday kitaplara CF skoru atanır.
* Kullanıcının daha önce etkileşime girdiği kitaplar öneri listesinden çıkarılır.

### Content-Based Filtering

İçerik tabanlı filtreleme bileşeni kitap metadata’sından türetilen metinsel temsilleri kullanır.

Her kitap için şu alanlar birleştirilerek `product_text` oluşturulur:

* Başlık
* Açıklama
* Kategori
* Yazar / yayıncı bilgisi
* Stil, kullanım ve materyal gibi yardımcı alanlar

TF-IDF ayarları:

```text
max_features = 10000
ngram_range = (1, 2)
sublinear_tf = True
matrix_format = sparse CSR
```

Kullanıcı profili, kullanıcının geçmişte etkileşim verdiği kitapların TF-IDF vektörlerinden rating ağırlıklı olarak oluşturulur.

### Semantic Retrieval

Semantik geri çağırma bileşeni, Türkçe kullanıcı sorguları ile çoğunlukla İngilizce olan kitap metadata’sını aynı vektör uzayında karşılaştırmak için kullanılır.

Kullanılan embedding modeli:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Her kitap için `product_text` alanı 384 boyutlu normalize embedding vektörüne dönüştürülür. Kullanıcı sorgusu da aynı modelle embedding’e çevrilir. Normalize vektörler üzerinde nokta çarpımı kullanılarak kosinüs benzerliği hesaplanır.

### Popularity Scoring

Popülerlik skoru, kitabın etkileşim sayısı ve ortalama puanı üzerinden hesaplanır.

```text
PopularityScore =
0.70 * normalized_interaction_count
+ 0.30 * normalized_average_rating
```

Popülerlik, hibrit sıralamada yardımcı sinyal olarak kullanılır.

### Category / Intent Matching

Tür/niyet eşleşmesi, Türkçe veya İngilizce kullanıcı sorgusundan kitap türü niyetini çıkarmak için kullanılır.

Örnek eşleşmeler:

| Sorgu            | Eşleşen kategori   |
| ---------------- | ------------------ |
| bilim kurgu      | Science Fiction    |
| romantik roman   | Romance            |
| korku romanı     | Horror             |
| polisiye         | Mystery & Thriller |
| psikoloji kitabı | Psychology         |
| kişisel gelişim  | Self-Help          |
| yemek kitabı     | Cooking            |

Bu bileşen, kullanıcının açık tür tercihinin hibrit skorlamaya dahil edilmesini sağlar.

---

## Hibrit Skor Formülleri

### Kişiselleştirilmiş Öneri

Geçmiş etkileşim verisi bulunan kullanıcılar için beş sinyal bileşeni birlikte kullanılır.

```text
HybridScore =
0.25 * CF
+ 0.20 * CBF
+ 0.35 * SEM
+ 0.10 * POP
+ 0.10 * CAT
```

Burada:

* `CF`: Collaborative Filtering skoru
* `CBF`: Content-Based Filtering skoru
* `SEM`: Semantic Similarity skoru
* `POP`: Popularity skoru
* `CAT`: Category / Intent Match skoru

Ağırlık toplamı 1.00’dır.

### Yeni Kullanıcı / Cold-Start Önerisi

Geçmiş etkileşim verisi olmayan kullanıcılar için CF ve CBF bileşenleri devre dışı bırakılır.

```text
NewUserScore =
0.70 * SEM
+ 0.20 * CAT
+ 0.10 * POP
```

Bu yapı, yeni kullanıcı önerilerinde doğal dil tercihi, tür/niyet eşleşmesi ve popülerlik sinyalini temel alır.

---

## Açıklama Üretimi

Sistemde iki seviyeli açıklama üretimi vardır.

### Kısa Gerekçe

Kısa gerekçe, büyük dil modeli kullanılmadan üretilir. Hibrit skor hesaplamasından sonra her öneri için baskın sinyal belirlenir ve bu sinyale karşılık gelen doğal Türkçe açıklama oluşturulur.

Örnek baskın sinyaller:

* `collaborative`
* `content`
* `semantic`
* `popularity`
* `category`

### Ayrıntılı Açıklama

Kullanıcı öneri kartında **Ayrıntılı açıklama oluştur** butonuna bastığında Qwen modeli devreye girer.

Kullanılan model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Modelin görevi öneri üretmek değildir. Öneri motoru tarafından seçilmiş gerçek katalog kitabı için, önerinin hangi sinyale dayandığını doğal Türkçe ile açıklamaktır.

Ayrıntılı açıklama üretiminde modele verilen temel bilgiler:

* Kitap adı
* Kitap kategorisi
* Kullanıcı sorgusu veya bağlamı
* Baskın öneri sinyali
* Kısa gerekçe

Uzun Amazon açıklamaları prompt içine doğrudan eklenmez. Bu tercih, küçük ölçekli modelin uzun İngilizce bağlam verildiğinde Türkçe açıklama kalitesinin düşebilmesi nedeniyle yapılmıştır.

---

## Proje Yapısı

```text
hybrid-recommender/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   ├── main.py
│   └── services/
│
├── frontend/
│   └── streamlit_app.py
│
├── recommender/
│   ├── collaborative.py
│   ├── content_based.py
│   ├── engine.py
│   ├── explainer.py
│   ├── profile.py
│   ├── scoring.py
│   ├── semantic.py
│   └── text_processing.py
│
├── scripts/
│   ├── download_amazon_subset.py
│   └── build_embeddings.py
│
├── evaluation/
│   └── metrics.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_text_processing.py
│   └── test_hybrid_weights.py
│
├── data/
│   ├── processed/
│   └── embeddings/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Kurulum

Aşağıdaki adımlar Windows ortamı içindir.

### 1. Depoyu klonlayın

```bash
git clone <repo-url>
cd hybrid-recommender
```

### 2. Sanal ortam oluşturun

```bash
python -m venv venv
```

### 3. Sanal ortamı etkinleştirin

```bash
venv\Scripts\activate
```

### 4. Paketleri yükleyin

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Ortam değişkenlerini oluşturun

```bash
copy .env.example .env
```

Linux / macOS için karşılık gelen komutlar:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---

## Veri Hazırlama

Amazon Reviews 2023 Books alt kümesini indirmek ve işlemek için:

```bash
python scripts/download_amazon_subset.py --category Books --max-products 1000 --max-interactions 10000 --min-user-interactions 3 --min-item-interactions 3
```

Bu işlem tamamlandığında aşağıdaki dosyalar oluşur:

```text
data/processed/products.csv
data/processed/interactions.csv
data/processed/users.csv
```

Notlar:

* İlk veri indirme işlemi internet hızına ve Hugging Face erişimine bağlı olarak uzun sürebilir.
* Eğer `data/processed/` klasörü içinde bu dosyalar zaten varsa, veri indirme adımı tekrar çalıştırılmak zorunda değildir.

---

## Embedding Oluşturma

İşlenmiş veri dosyaları hazırlandıktan sonra kitap embeddinglerini oluşturmak için:

```bash
python scripts/build_embeddings.py
```

Bu işlem tamamlandığında aşağıdaki dosyalar oluşur:

```text
data/embeddings/product_embeddings.npy
data/embeddings/product_ids.txt
```

Embeddingler bir kez oluşturulduktan sonra uygulama sonraki çalıştırmalarda bu dosyaları doğrudan kullanır.

---

## Uygulamayı Çalıştırma

Uygulama iki ayrı terminal ile çalıştırılır.

### 1. Backend başlatma

Birinci terminalde:

```bash
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

Backend çalıştıktan sonra API dokümantasyonuna şu adresten erişilebilir:

```text
http://localhost:8000/docs
```

Health kontrolü:

```text
http://localhost:8000/health
```

### 2. Frontend başlatma

İkinci terminalde:

```bash
venv\Scripts\activate
streamlit run frontend/streamlit_app.py
```

Windows’ta alternatif olarak şu komut da kullanılabilir:

```bash
streamlit run frontend\streamlit_app.py
```

Streamlit arayüzü varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

---

## Kullanım Rehberi

### Kişisel Öneriler

1. **Kişisel Öneriler** sekmesine girin.
2. Listeden bir okuyucu seçin.
3. İsteğe bağlı olarak arama ifadesi yazın.

   * Örnek: `korku romanı`, `bilim kurgu`, `psikoloji kitabı`
4. **Kitap Öner** butonuna basın.
5. Sistem, kullanıcının geçmiş okuma davranışı ve sorgu bağlamına göre öneriler üretir.

### Yeni Okuyucu

1. **Yeni Okuyucu** sekmesine girin.
2. Kullanıcı adını yazın.
3. Ne aradığınızı doğal dille belirtin.

   * Örnek: `kişisel gelişim kitabı arıyorum`
4. **Bana Kitap Öner** butonuna basın.
5. Sistem, geçmiş kullanıcı verisi olmadan semantik benzerlik, tür/niyet eşleşmesi ve popülerlik sinyalleriyle öneri üretir.

### Kitap Arama

1. **Kitap Arama** sekmesine girin.
2. Türkçe doğal dil sorgusu yazın.

   * Örnek: `yemek kitabı`, `fantastik roman`, `polisiye roman`
3. **Ara** butonuna basın.
4. Sistem, sorgu ile kitap metadata’sı arasındaki anlamsal ve kategorik uyumu kullanarak sonuçları listeler.

### Benzer Kitaplar

Öneri kartlarında yer alan **Benzer kitapları göster** butonu ile seçilen kitaba benzer kitaplar görüntülenebilir. Bu işlemde semantik benzerlik ve içerik tabanlı benzerlik birlikte kullanılır.

### Kitap Karşılaştırma

**Karşılaştırma** sekmesinde iki veya üç kitap seçilerek temel katalog bilgileri tablo halinde karşılaştırılabilir. Bu bölüm karar desteği sağlamak amacıyla kitapların kategori, yazar/yayıncı, fiyat, puan ve açıklama gibi alanlarını yan yana gösterir.

### Sistem Bilgisi ve Değerlendirme

**Sistem Bilgisi** sekmesinde:

* Kullanılan modeller,
* Veri seti özeti,
* Öneri skoru mantığı,
* Performans bilgileri,
* Top-K değerlendirme metrikleri

görüntülenebilir.

---

## API Uç Noktaları

| Endpoint                   | Yöntem | Açıklama                                              |
| -------------------------- | ------ | ----------------------------------------------------- |
| `/`                        | GET    | Servis bilgisi                                        |
| `/health`                  | GET    | Sistem sağlık kontrolü                                |
| `/users`                   | GET    | Kullanıcı listesini döndürür                          |
| `/users/{user_id}/profile` | GET    | Kullanıcı profil özetini döndürür                     |
| `/recommend`               | GET    | Mevcut kullanıcı için kişiselleştirilmiş öneri üretir |
| `/recommend/new-user`      | GET    | Yeni kullanıcı / cold-start önerisi üretir            |
| `/search`                  | GET    | Doğal dil sorgusuyla kitap arama yapar                |
| `/similar/{product_id}`    | GET    | Seçilen kitaba benzer kitapları döndürür              |
| `/explain`                 | POST   | Seçilen öneri için ayrıntılı açıklama üretir          |
| `/compare`                 | POST   | Seçilen kitapların katalog bilgilerini karşılaştırır  |
| `/metrics/system`          | GET    | Sistem ve veri özeti bilgilerini döndürür             |
| `/metrics/latency`         | GET    | Ortalama gecikme bilgilerini döndürür                 |
| `/metrics/evaluation`      | GET    | Top-K değerlendirme metriklerini hesaplar             |
| `/model-info`              | GET    | Kullanılan model ve bileşen bilgilerini döndürür      |

---

## Değerlendirme

Sistem, Top-K öneri metrikleriyle değerlendirilmektedir.

Kullanılan metrikler:

* Precision@K
* Recall@K
* NDCG@K
* HitRate@K

Değerlendirme sürecinde temporal holdout yaklaşımı kullanılır. Uygun kullanıcılar için kronolojik olarak en güncel 2–3 etkileşim test kümesine ayrılır. Kalan geçmiş etkileşimler öneri motorunun kurulması ve öneri üretimi için kullanılır. Bu yaklaşım, test verisinin öneri üretim sürecine sızmasını azaltmak amacıyla tercih edilmiştir.

Örnek değerlendirme isteği:

```text
http://localhost:8000/metrics/evaluation?k=10&max_users=50
```

Streamlit arayüzünden de **Sistem Bilgisi > Öneri Kalite Değerlendirmesi** bölümünden değerlendirme çalıştırılabilir.

---

## Testler

Tüm testleri çalıştırmak için:

```bash
python -m pytest tests/ -q
```

Beklenen örnek çıktı:

```text
33 passed
```

Test kapsamı:

* Precision@K, Recall@K, NDCG@K ve HitRate@K hesaplamaları
* Temporal holdout değerlendirme akışı
* Türkçe kitap türü / niyet algılama
* Hibrit skor ağırlıklarının toplamı
* Cold-start ağırlıklarının toplamı
* Temel metin işleme yardımcıları

---

## Çevre Değişkenleri

`.env` dosyasında kullanılabilecek temel değişkenler:

```env
AMAZON_CATEGORY=Books

EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

LLM_PROVIDER=huggingface
LLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
HF_TOKEN=your_huggingface_read_token
HF_MAX_NEW_TOKENS=90

WEIGHT_COLLABORATIVE=0.25
WEIGHT_CONTENT=0.20
WEIGHT_SEMANTIC=0.35
WEIGHT_POPULARITY=0.10
WEIGHT_CATEGORY_INTENT=0.10

WEIGHT_COLD_SEMANTIC=0.70
WEIGHT_COLD_CATEGORY=0.20
WEIGHT_COLD_POPULARITY=0.10
```

Notlar:

* `HF_TOKEN`, Hugging Face veri seti veya model erişiminde gerekebilir.
* Qwen açıklaması CPU ortamında yavaş çalışabilir.
* Qwen modeli yalnızca ayrıntılı açıklama istenirse devreye girer.
* Hızlı test için açıklama sağlayıcısı template moduna alınabilir:

```env
LLM_PROVIDER=template
```

---
## Sınırlılıklar

* Sistem Amazon Reviews 2023 Books veri setinin yönetilebilir bir alt kümesi üzerinde çalışmaktadır.
* Kullanıcı başına etkileşim sayısı düşük olduğunda collaborative filtering bileşeni sınırlı davranışsal sinyal üretebilir.
* Amazon Books metadata’sındaki kategori alanı bazı kitapların alt türünü veya tematik içeriğini tam olarak yansıtmayabilir.
* Türkçe sorgular ile İngilizce metadata arasındaki anlamsal eşleşme her sorguda kusursuz sonuç üretmeyebilir.
* Qwen2.5-0.5B-Instruct küçük ölçekli bir modeldir; bazı açıklamalarda dilsel akıcılık veya tekrar sorunları oluşabilir.
* CPU ortamında LLM tabanlı ayrıntılı açıklama üretimi görece yavaş çalışabilir.
* Sistem gerçek kullanıcılarla yapılmış çevrimiçi A/B test içermemektedir.

---

## Lisans ve Kullanım Notu

Bu proje araştırma, eğitim ve prototip geliştirme amacıyla hazırlanmıştır. Amazon Reviews 2023 veri seti ve kullanılan modeller kendi lisans ve kullanım koşullarına tabidir.


