"""
Açıklama Servisi

Tasarım kararı:
- Öneriyi Qwen üretmez; öneriyi hibrit öneri motoru üretir.
- Bu servis, hibrit motorun hesapladığı baskın sinyali ve ürün/kullanıcı bağlamını
  kullanıcıya doğal Türkçe ile aktarır.
- Hızlı modda dinamik template açıklaması üretir.
- LLM_PROVIDER=huggingface ise, yalnızca kullanıcı ayrıntılı açıklama isterse Qwen
  ile metni daha doğal ve ayrıntılı hale getirir.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Qwen destekli, sinyal temelli açıklama servisi."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_loaded: bool = False
        self._load_error: Optional[str] = None

    # ──────────────────────────────────────────
    # Yardımcı metin fonksiyonları
    # ──────────────────────────────────────────

    @staticmethod
    def _short(text: str, limit: int = 180) -> str:
        text = str(text or "").replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _query_phrase(user_query: str) -> str:
        q = str(user_query or "").strip()
        return f"“{q}”" if q else "mevcut okuma bağlamınız"

    @staticmethod
    def _clean_category(category: str) -> str:
        category = str(category or "kitap").strip()
        return category if category and category.lower() != "nan" else "kitap"

    @staticmethod
    def _sentence_join(sentences: list[str], max_sentences: int = 4) -> str:
        clean = []
        for s in sentences:
            s = str(s or "").strip()
            if not s:
                continue
            if not s.endswith((".", "!", "?")):
                s += "."
            clean.append(s)
        return " ".join(clean[:max_sentences]).strip()

    # ──────────────────────────────────────────
    # Sinyal temelli gerekçe üretimi
    # ──────────────────────────────────────────

    def natural_reasons(
        self,
        product_title: str,
        product_description: str,
        product_category: str,
        user_query: str = "",
        dominant_signal: str = "semantic",
    ) -> list[str]:
        """
        Hibrit öneri motorundaki sinyalleri kullanıcı dilindeki doğal gerekçelere dönüştürür.

        Not:
        - Burada raw kitap açıklaması uzun biçimde verilmez.
        - Amaç kitabın konusunu özetlemek değil, öneri kararının dayandığı sinyalleri açıklamaktır.
        """
        title = self._short(product_title, 120)
        category = self._clean_category(product_category)
        query = self._query_phrase(user_query)

        reason_map = {
            "collaborative": [
                "geçmiş okuma davranışınızla benzer örüntü gösteren kullanıcıların bu kitaba yakın adaylarla ilgilenmiş olması",
                f"{category} türündeki önceki etkileşimlerinizin bu kitabı kişiselleştirilmiş liste için güçlü bir seçenek haline getirmesi",
                "kullanıcı davranışına dayalı benzerlik sinyalinin öneri sıralamasında etkili olması",
            ],
            "content": [
                "kitabın başlık, tür ve katalog bilgilerinin daha önce ilgi gösterilen kitaplarla içerik bakımından benzerlik taşıması",
                f"{category} türü ve kitap bilgilerinin okuma profilinizdeki içerik eğilimiyle örtüşmesi",
                "içerik tabanlı benzerliğin bu kitabı benzer okuma beklentilerine yakınlaştırması",
            ],
            "semantic": [
                f"kitabın {query} ifadesindeki okuma niyetiyle anlam bakımından uyumlu görülmesi",
                "başlık, tür ve katalog bilgilerinin arama tercihinize yakın bir okuma deneyimine işaret etmesi",
                "doğal dilde yazılan tercihin kitap kataloğundaki anlamca yakın adaylarla eşleştirilmesi",
            ],
            "category": [
                f"kitabın aranan konuya yakın {category} türünde yer alması",
                "tür bilgisinin kullanıcı tercihi ile katalogdaki kitaplar arasında doğrudan bağlantı kurması",
                "tema ve tür uyumunun öneri sıralamasında belirleyici sinyallerden biri olması",
            ],
            "popularity": [
                "kitabın katalog içinde kullanıcıların sık etkileşim gösterdiği seçeneklerden biri olması",
                "genel kullanıcı ilgisinin kitabın öneri sıralamasındaki konumunu desteklemesi",
                "popülerlik sinyalinin içerik ve tercih uyumuyla birlikte değerlendirilmesi",
            ],
        }

        return reason_map.get(dominant_signal, reason_map["semantic"])

    def quick_explain(
        self,
        product_title: str,
        product_description: str,
        product_category: str,
        user_query: str = "",
        dominant_signal: str = "semantic",
        mode: str = "personalized",
    ) -> str:
        """LLM yüklemeden hızlı, dinamik ve sinyal temelli kısa açıklama üretir."""
        reasons = self.natural_reasons(
            product_title=product_title,
            product_description=product_description,
            product_category=product_category,
            user_query=user_query,
            dominant_signal=dominant_signal,
        )

        if mode == "cold_start":
            intro = (
                "Geçmiş etkileşim olmadığı için bu öneri, yazdığınız tercih ile kitap kataloğu "
                "arasındaki anlam uyumuna göre oluşturuldu."
            )
            if dominant_signal == "popularity":
                intro = (
                    "Geçmiş etkileşim olmadığı için sistem, tercih metninizi katalogdaki genel ilgi "
                    "düzeyiyle birlikte değerlendirdi."
                )
        elif not user_query and dominant_signal == "collaborative":
            intro = "Bu öneri, önceki okuma davranışınızdan çıkarılan kişiselleştirilmiş sinyallere dayanıyor."
        else:
            intro = "Bu öneri, seçilen kitabın katalog bilgileri ile kullanıcı bağlamı birlikte değerlendirilerek oluşturuldu."

        return f"{intro} Bu kitap, {reasons[0]} nedeniyle öne çıkarıldı."

    def detailed_template_explain(
        self,
        product_title: str,
        product_description: str,
        product_category: str,
        user_query: str = "",
        dominant_signal: str = "semantic",
    ) -> str:
        """
        Qwen başarısız olursa kullanılacak güvenli ayrıntılı açıklama.

        Bu açıklama kitap özeti değildir; öneri kararının nedenini 3-4 cümleyle açıklar.
        """
        category = self._clean_category(product_category)
        query = self._query_phrase(user_query)
        reasons = self.natural_reasons(
            product_title=product_title,
            product_description=product_description,
            product_category=product_category,
            user_query=user_query,
            dominant_signal=dominant_signal,
        )

        if dominant_signal == "collaborative":
            sentences = [
                "Bu kitap, önceki okuma davranışınızla ilişkili kişiselleştirilmiş sinyaller dikkate alınarak önerildi",
                "Sistem, benzer okuma örüntülerine sahip kullanıcıların ilgilendiği adayları ve sizin geçmiş etkileşimlerinizi birlikte değerlendirdi",
                f"{category} türüyle ilişkili katalog bilgileri de bu kitabın okuma profilinize yakın bir seçenek olarak öne çıkmasını destekledi",
            ]
        elif dominant_signal == "content":
            sentences = [
                "Bu kitap, daha önce ilgi gösterdiğiniz kitaplarla içerik ve tür bakımından benzer özellikler taşıdığı için önerildi",
                "Sistem, kitabın başlık, tür ve katalog bilgilerini geçmiş okuma profilinizle karşılaştırdı",
                "Bu benzerlik, kitabı aynı okuma beklentisine yakın adaylar arasında daha güçlü bir seçenek haline getirdi",
            ]
        elif dominant_signal == "category":
            sentences = [
                f"Bu kitap, aradığınız konuya yakın {category} türünde yer aldığı için önerildi",
                "Sistem, tür bilgisini kullanıcı tercihiyle birlikte değerlendirerek katalogdaki uygun adayları öne çıkardı",
                "Bu nedenle kitap, arama niyetinizle doğrudan ilişkili seçeneklerden biri olarak sıralamaya dahil edildi",
            ]
        elif dominant_signal == "popularity":
            sentences = [
                "Bu kitap, katalog içinde kullanıcıların sık etkileşim gösterdiği seçeneklerden biri olduğu için önerildi",
                "Sistem, genel kullanıcı ilgisini kitap bilgileri ve kullanıcı bağlamıyla birlikte değerlendirdi",
                "Bu nedenle kitap, yalnızca popüler olduğu için değil, mevcut okuma ihtiyacınıza yakın göründüğü için de listeye dahil edildi",
            ]
        else:
            sentences = [
                f"Bu kitap, {query} ifadesindeki okuma niyetiyle anlam bakımından uyumlu görüldüğü için önerildi",
                "Sistem, yazdığınız tercihi kitabın başlık, tür ve katalog bilgileriyle karşılaştırdı",
                "Bu karşılaştırma sonucunda kitap, aradığınız okuma deneyimine yakın adaylardan biri olarak değerlendirildi",
            ]

        return self._sentence_join(sentences, max_sentences=4)

    # ──────────────────────────────────────────
    # Qwen yükleme ve chat-template üretimi
    # ──────────────────────────────────────────

    def _load_model(self) -> bool:
        """Qwen modelini lazy-loading ile yükler."""
        if settings.llm_provider.lower() != "huggingface":
            self._load_error = "LLM_PROVIDER template olarak ayarlı."
            return False
        if self._model_loaded:
            return True
        if self._load_error:
            return False

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_name = settings.llm_model_name
            token = settings.hf_token.strip() or None
            logger.info(f"LLM yükleniyor: {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=token,
                trust_remote_code=True,
            )
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as first_error:
                logger.warning(f"device_map='auto' başarısız; CPU fallback deneniyor: {first_error}")
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

            # Qwen generation_config içinde sampling parametreleri varsayılan gelebiliyor.
            # Deterministik açıklama üretimi için do_sample=False kullanıyoruz.
            # Bu nedenle sampling parametrelerini temizleyip uyarıları önlüyoruz.
            try:
                self._model.generation_config.do_sample = False
                self._model.generation_config.temperature = None
                self._model.generation_config.top_p = None
                self._model.generation_config.top_k = None
            except Exception:
                pass

            self._model.eval()
            self._model_loaded = True
            logger.info("LLM başarıyla yüklendi.")
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.error(f"LLM yüklenemedi: {exc}")
            return False

    def _build_messages(
        self,
        product_title: str,
        product_description: str,
        product_category: str,
        user_query: str,
        dominant_signal: str,
    ) -> list[dict]:
        """
        Qwen için sade ve sınırlı prompt hazırlar.

        Önemli:
        - Uzun kitap açıklaması prompta verilmez.
        - Modelden kitap özeti değil, önerinin nedenini açıklaması istenir.
        - Çıktıda başlık, madde veya prompt etiketi istenmez.
        """
        reasons = self.natural_reasons(
            product_title=product_title,
            product_description=product_description,
            product_category=product_category,
            user_query=user_query,
            dominant_signal=dominant_signal,
        )
        short_reason = self.quick_explain(
            product_title=product_title,
            product_description=product_description,
            product_category=product_category,
            user_query=user_query,
            dominant_signal=dominant_signal,
        )
        category = self._clean_category(product_category)
        query = str(user_query or "kişiselleştirilmiş okuma önerisi").strip()

        user_content = (
            f"Kullanıcı tercihi: {query}\n"
            f"Önerilen kitap: {self._short(product_title, 120)}\n"
            f"Kitap türü: {category}\n"
            f"Sistemin kısa gerekçesi: {short_reason}\n"
            f"Öneriyi destekleyen doğal nedenler: {', '.join(reasons[:2])}\n\n"
            "Bu bilgilerle, bu kitabın neden önerildiğini kullanıcıya doğal Türkçe ile açıkla. "
            "Kitabın konusunu özetleme. Kısa gerekçeyi aynen tekrar etme. "
            "Madde işareti, başlık veya etiket kullanma. "
            "Sadece 3 veya 4 cümlelik nihai açıklamayı yaz."
        )

        return [
            {
                "role": "system",
                "content": (
                    "Sen bir kitap öneri sisteminde açıklama yazan yardımcı modelsin. "
                    "Öneriyi sen üretmezsin; yalnızca sistemin seçtiği gerçek kitabın neden önerildiğini açıklarsın. "
                    "Kitap özeti veya kitap tanıtımı yazma. "
                    "Sadece Türkçe yaz. İngilizce cümle kurma. "
                    "Prompttaki alan adlarını, başlıkları veya etiketleri çıktıya yazma. "
                    "Teknik skor, kod, formül veya değişken adı kullanma. "
                    "Açıklama 3 veya 4 doğal cümleden oluşsun."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_comparison_messages(self, products: list[dict], usage_context: str) -> list[dict]:
        lines = []
        for i, p in enumerate(products, 1):
            lines.append(
                f"Kitap {i}: {p.get('title', '')} | Tür: {p.get('category', '')} | "
                f"Yazar/Yayıncı: {p.get('brand', '')} | Fiyat: {p.get('price_tl', p.get('price', ''))} | "
                f"Açıklama: {self._short(p.get('description', ''), 260)}"
            )
        return [
            {
                "role": "system",
                "content": (
                    "Sen bir kitap karşılaştırma yardımcısısın. Sadece verilen kitapları karşılaştır. "
                    "Yeni kitap, yazar veya özellik uydurma. Sade Türkçe kullan."
                ),
            },
            {
                "role": "user",
                "content": f"""
Okuma amacı: {usage_context if usage_context else "genel okuma tercihi"}

Karşılaştırılacak kitaplar:
{chr(10).join(lines)}

Görev: Bu kitapları okuma amacı, tür, açıklama ve fiyat bilgilerine göre 3 kısa cümleyle karşılaştır.
""".strip(),
            },
        ]

    def _generate_chat(self, messages: list[dict], max_new_tokens: int = 140) -> str:
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ──────────────────────────────────────────
    # Çıktı temizleme ve güvenlik kontrolü
    # ──────────────────────────────────────────

    @staticmethod
    def _looks_bad_output(text: str) -> bool:
        """Qwen promptu kopyaladıysa, İngilizceye kaydıysa veya çok kısa kaldıysa yakalar."""
        raw = str(text or "").strip()
        lower = raw.lower()

        if len(raw) < 120:
            return True

        bad_patterns = [
            "kullanıcı bağlamı",
            "kullanıcı tercihi:",
            "gerçek katalog kitabı",
            "önerilen kitap:",
            "kitap adı:",
            "kitap türü:",
            "tür/kategori:",
            "sistemin kısa gerekçesi",
            "öneriyi destekleyen",
            "görev:",
            "bu kitabın neden önerildiğini sadeleştir",
            "sadeleştirilmiş türkçe",
            "review praise",
            "user context",
            "recommendation reason",
            "the book",
            "this book is recommended because",
            "- ",
            "•",
        ]

        return any(pattern in lower for pattern in bad_patterns)

    @staticmethod
    def _clean_output(text: str, max_sentences: int = 4) -> str:
        text = str(text or "").strip()

        # Olası başlık/etiket tekrarlarını temizle.
        text = re.sub(r"(?i)^(açıklama|cevap|yanıt|sonuç)\s*:\s*", "", text).strip()
        text = re.sub(r"(?i)^bu kitabın neden önerildiğini sadeleştirerek\s*:\s*", "", text).strip()
        text = re.sub(r"(?i)^sadeleştirilmiş türkçe çevirisi\s*:\s*", "", text).strip()

        # Satır satır filtrele: prompt etiketleri ve maddeler atılır.
        blocked_starts = (
            "kullanıcı bağlamı",
            "kullanıcı tercihi",
            "gerçek katalog kitabı",
            "önerilen kitap",
            "kitap adı",
            "kitap türü",
            "tür/kategori",
            "öneriyi destekleyen",
            "sistemin kısa gerekçesi",
            "görev",
        )

        lines = []
        for line in text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            lowered = clean_line.lower()
            if lowered.startswith(blocked_starts):
                continue
            if clean_line.startswith(("-", "•", "*")):
                continue
            lines.append(clean_line)

        text = " ".join(lines).strip()
        text = re.sub(r"\s+", " ", text)

        forbidden = [
            "semantic_score", "cf_score", "content_score", "hybrid_score",
            "popularity_score", "category_match", "skor=", "score=",
            "Collaborative Filtering", "Content-Based", "RAG",
        ]
        for token in forbidden:
            text = text.replace(token, "")

        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = " ".join(sentences[:max_sentences]).strip()

        return result if result else text[:450]

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def explain(
        self,
        product_title: str,
        product_description: str,
        product_category: str,
        user_query: str = "",
        dominant_signal: str = "semantic",
    ) -> str:
        """Qwen açıksa Qwen ile, değilse güvenli template ile ayrıntılı açıklama üretir."""
        fallback = self.detailed_template_explain(
            product_title=product_title,
            product_description=product_description,
            product_category=product_category,
            user_query=user_query,
            dominant_signal=dominant_signal,
        )

        if settings.llm_provider.lower() != "huggingface":
            return fallback

        if not self._model_loaded and not self._load_error:
            self._load_model()

        if not self._model_loaded:
            logger.warning("LLM mevcut değil, sinyal temelli ayrıntılı template açıklama kullanılıyor.")
            return fallback

        try:
            messages = self._build_messages(
                product_title=product_title,
                product_description=product_description,
                product_category=product_category,
                user_query=user_query,
                dominant_signal=dominant_signal,
            )
            text = self._generate_chat(
                messages,
                max_new_tokens=getattr(settings, "hf_max_new_tokens", 140),
            )
            cleaned = self._clean_output(text, max_sentences=4)

            if self._looks_bad_output(cleaned):
                return fallback

            return cleaned

        except Exception as exc:
            logger.error(f"LLM açıklama hatası: {exc}", exc_info=True)
            return fallback

    def compare(self, products: list[dict], usage_context: str = "") -> str:
        """Karşılaştırma yorumu üretir; Qwen yoksa ürünlere göre dinamik template kullanır."""
        if settings.llm_provider.lower() == "huggingface":
            if not self._model_loaded and not self._load_error:
                self._load_model()
            if self._model_loaded:
                try:
                    messages = self._build_comparison_messages(products, usage_context)
                    return self._clean_output(self._generate_chat(messages, max_new_tokens=120), max_sentences=4)
                except Exception as exc:
                    logger.error(f"LLM karşılaştırma hatası: {exc}", exc_info=True)

        if len(products) < 2:
            return "Karşılaştırma için yeterli kitap bilgisi bulunamadı."

        p1, p2 = products[0], products[1]
        purpose = f" {usage_context} amacıyla" if usage_context else ""
        same_cat = str(p1.get("category", "")) == str(p2.get("category", ""))

        if same_cat:
            return (
                f"Bu iki kitap aynı türe yakın olduğu için{purpose} doğrudan karşılaştırılabilir. "
                f"{p1.get('title','İlk kitap')} açıklama ve fiyat dengesiyle, "
                f"{p2.get('title','ikinci kitap')} ise benzer türde alternatif okuma deneyimiyle öne çıkar."
            )

        return (
            f"Bu kitaplar farklı okuma beklentilerine hitap ediyor. "
            f"{p1.get('title','İlk kitap')} daha çok {p1.get('category','belirsiz tür')} yönüne, "
            f"{p2.get('title','ikinci kitap')} ise {p2.get('category','belirsiz tür')} yönüne yakındır."
        )


llm_explainer = LLMExplainer()