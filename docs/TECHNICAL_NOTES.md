# Teknik Notlar

## Mimari karar

Sistemde Qwen doğrudan kitap önermez. Öneri üretimi şu hibrit yapıyla yapılır:

1. Item-Based Collaborative Filtering
2. TF-IDF Content-Based Filtering
3. SentenceTransformer tabanlı semantik retrieval
4. Popülerlik skoru
5. Türkçe sorgular için tür/niyet eşleştirme

Qwen yalnızca açıklama katmanında kullanılır. Kullanıcı “Ayrıntılı açıklama oluştur” dediğinde seçilmiş gerçek katalog kitabı, kitap açıklaması ve öneriyi destekleyen doğal gerekçeler Qwen’e verilir. Bu yaklaşım katalog dışı kitap uydurma riskini azaltır.

## Açıklama katmanı

Her öneri için baskın sinyal belirlenir:

- collaborative: kullanıcı davranışı / benzer okuma örüntüsü
- content: kitap içeriği, tür ve açıklama benzerliği
- semantic: doğal dil sorgusu ile kitap metni arasındaki anlamsal uyum
- category: tür/konu niyeti eşleşmesi
- popularity: katalogdaki genel kullanıcı ilgisi

Qwen kapalıyken de sistem bu sinyallere dayalı dinamik template açıklaması üretir. Bu nedenle açıklamalar sabit cümlelerden oluşmaz; önerinin dayandığı sinyale göre değişir.

## Performans kararı

Qwen CPU ortamında yavaş çalışabildiği için öneri listesi oluşturulurken LLM çağrısı yapılmaz. Öneriler hızlı gelir; LLM yalnızca kullanıcı talep ettiğinde açıklama üretir. Bu karar hem kullanıcı deneyimini hem de deploy maliyetini iyileştirir.

## Cold-start

Yeni kullanıcıda geçmiş etkileşim olmadığı için CF skoru kullanılamaz. Bu durumda tercih metni, kitap açıklamalarıyla semantik olarak eşleştirilir; tür/konu niyeti ve popülerlik yardımcı sinyal olarak kullanılır.

## Son düzenleme: sinyal tabanlı açıklama ve tür niyeti filtresi

Kullanıcı arayüzünde teknik terimler gösterilmez; ancak backend her öneri için gerçek öneri sinyallerini hesaplar. Sistem önce hibrit motorla kitabı seçer, sonra baskın sinyale göre kısa doğal gerekçe üretir. Qwen aktifse yalnızca kullanıcı ayrıntılı açıklama istediğinde çalışır ve bu sinyalleri doğal Türkçe açıklamaya dönüştürür.

Books veri setinde kategori alanı çoğu zaman `Fiction` gibi geniş üst türler içerdiği için, arama niyeti yalnızca kategori adıyla değil; kitap başlığı, açıklama, product_text ve tür eş anlamlılarıyla da eşleştirilir. Örneğin `romantik roman` sorgusu `Romance`, `korku romanı` sorgusu `Horror`, `bilim kurgu` sorgusu `Science Fiction` niyetine dönüştürülür.

## Değerlendirme Protokolü Güncellemesi

Değerlendirme aşamasında veri sızıntısını önlemek için kullanıcı etkileşimleri zamana göre ayrılır. Her uygun kullanıcı için son 2–3 etkileşim test kümesi olarak tutulur; öneri motoru ise yalnızca kalan train etkileşimleriyle yeniden kurulur. Bu nedenle CF eğitimi ve CBF kullanıcı profil hesabı test öğelerini görmez. Tek öğeli klasik leave-one-out yerine çoklu temporal holdout kullanılması, Recall@K ile HitRate@K metriklerinin zorunlu olarak aynı değere düşmesini de azaltır.

## Ölçeklenebilirlik Güncellemesi

TF-IDF matrisi sparse CSR formatında saklanır ve içerik benzerliği hesaplamalarında `product_id → index` sözlüğü kullanılır. Böylece dense matris kaynaklı bellek tüketimi ve `product_ids.index(pid)` kaynaklı O(n) arama maliyeti önlenir. Collaborative filtering skorlaması da adaylar üzerinde vektörize matris işlemi ile hesaplanır.
