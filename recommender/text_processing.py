"""
Metin işleme yardımcıları.
Türkçe sorguları kitap türü eş anlamlılarıyla genişletir, kategori niyetini
algılar ve ürün metinlerini öneri bileşenleri için hazırlar.
"""

from __future__ import annotations

from typing import Optional

BOOK_GENRE_SYNONYMS: dict[str, list[str]] = {
    "Fiction": [
        "roman", "edebiyat", "kurgu", "hikaye", "öykü", "novel", "fiction",
        "literature", "story", "stories"
    ],
    "Science Fiction": [
        "bilim kurgu", "bilimkurgu", "sci-fi", "scifi", "science fiction",
        "uzay", "space", "dystopian", "dystopia"
    ],
    "Fantasy": [
        "fantastik", "fantasy", "büyü", "magic", "dragon", "epic fantasy"
    ],
    "Mystery & Thriller": [
        "polisiye", "gizem", "gerilim", "suç", "dedektif", "mystery",
        "thriller", "crime", "detective", "suspense"
    ],
    "Romance": [
        "romantik", "romantik roman", "aşk", "aşk romanı", "romance", "romantic",
        "love story", "love", "contemporary romance", "romantic suspense"
    ],
    "Horror": [
        "korku", "korku romanı", "horror", "haunted", "ghost", "supernatural",
        "zombie", "post-apocalyptic", "apocalyptic", "dark", "scary"
    ],
    "History": [
        "tarih", "tarihi", "history", "historical", "ottoman", "world war",
        "biography history"
    ],
    "Biography": [
        "biyografi", "otobiyografi", "anı", "memoir", "biography", "autobiography"
    ],
    "Psychology": [
        "psikoloji", "psychology", "mental health", "mind", "therapy", "behavior"
    ],
    "Self-Help": [
        "kişisel gelişim", "motivasyon", "öz gelişim", "self help", "self-help",
        "personal development", "productivity", "habit", "success"
    ],
    "Business": [
        "iş", "iş dünyası", "girişimcilik", "yönetim", "pazarlama", "business",
        "management", "entrepreneurship", "startup", "marketing", "finance"
    ],
    "Children": [
        "çocuk", "çocuk kitabı", "kids", "children", "juvenile", "picture book",
        "young readers"
    ],
    "Education": [
        "ders", "eğitim", "akademik", "textbook", "education", "academic",
        "study", "exam", "reference", "school"
    ],
    "Cooking": [
        "yemek", "tarif", "mutfak", "cookbook", "cooking", "recipe", "baking"
    ],
    "Religion": [
        "din", "maneviyat", "religion", "spirituality", "christian", "islam",
        "bible", "quran"
    ],
    "Comics": [
        "çizgi roman", "manga", "comic", "comics", "graphic novel", "anime"
    ],
    "Books": [
        "kitap", "kitap öner", "book", "books", "reading", "okuma"
    ],
}

GENERAL_SYNONYMS: dict[str, str] = {
    "beyaz": "white",
    "siyah": "black",
    "ucuz": "affordable budget low price",
    "uygun fiyat": "affordable budget",
    "yeni": "new recent modern",
    "popüler": "popular bestselling bestseller",
    "hediye": "gift present",
    "ingilizce": "english language",
    "türkçe": "turkish language",
    "başlangıç": "beginner introductory easy",
    "ileri": "advanced expert",
}


def normalize_text(text: str) -> str:
    return str(text or "").lower().strip()


def detect_category_intent(query: str) -> Optional[str]:
    """Türkçe/İngilizce sorgudan kitap türü niyeti algılar. """
    normalized = normalize_text(query)
    generic_categories = {"Fiction", "Books"}

    def score_category(category: str) -> float:
        score = 0.0
        for kw in BOOK_GENRE_SYNONYMS.get(category, []):
            kw_l = kw.lower()
            if kw_l in normalized:
                # Çok kelimeli / daha uzun ifadeler daha özgül kabul edilir.
                score += 1.0 + min(len(kw_l), 30) / 20.0
                if " " in kw_l:
                    score += 0.75
        return score

    specific_scores = {c: score_category(c) for c in BOOK_GENRE_SYNONYMS if c not in generic_categories}
    best_specific = max(specific_scores, key=specific_scores.get) if specific_scores else None
    if best_specific and specific_scores[best_specific] > 0:
        return best_specific

    generic_scores = {c: score_category(c) for c in generic_categories}
    best_generic = max(generic_scores, key=generic_scores.get) if generic_scores else None
    return best_generic if best_generic and generic_scores[best_generic] > 0 else None


def expand_query(query: str) -> str:
    """Türkçe sorguyu kitap türü/konu eş anlamlılarıyla genişletir."""
    normalized = normalize_text(query)
    expansions: list[str] = [query]

    for key, value in GENERAL_SYNONYMS.items():
        if key in normalized:
            expansions.append(value)

    for category, keywords in BOOK_GENRE_SYNONYMS.items():
        if any(kw.lower() in normalized for kw in keywords):
            expansions.append(category)
            english = [kw for kw in keywords if kw.isascii() and len(kw) > 3]
            expansions.extend(english[:6])

    seen = set()
    out = []
    for item in expansions:
        item = str(item).strip()
        if item and item.lower() not in seen:
            out.append(item)
            seen.add(item.lower())
    return " ".join(out)


def infer_book_category(text: str) -> str:
    """Metadata metninden kitap türü çıkarır. Bulamazsa Books döner."""
    return detect_category_intent(text) or "Books"


def genre_keywords(category: str | None) -> list[str]:
    """Tespit edilen tür için arama/filtreleme anahtarlarını döndürür."""
    if not category:
        return []
    return BOOK_GENRE_SYNONYMS.get(str(category), [])


def genre_match_score(text: str, detected_category: str | None) -> float:
    """Ürün metni ile tespit edilen tür niyeti arasındaki uyumu hesaplar."""
    if not detected_category:
        return 0.5
    normalized = normalize_text(text)
    det = str(detected_category).lower()
    if det and det in normalized:
        return 1.0
    keywords = genre_keywords(detected_category)
    hits = 0
    weighted = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l and kw_l in normalized:
            hits += 1
            weighted += 1.0 + min(len(kw_l), 24) / 24.0
    if hits >= 2 or weighted >= 2.3:
        return 1.0
    if hits == 1:
        return 0.75
    return 0.0


def readable_genre(category: str | None) -> str:
    mapping = {
        "Science Fiction": "Bilim kurgu",
        "Fantasy": "Fantastik",
        "Mystery & Thriller": "Polisiye / gerilim",
        "Romance": "Romantik roman",
        "Horror": "Korku",
        "Psychology": "Psikoloji",
        "Self-Help": "Kişisel gelişim",
        "History": "Tarih",
        "Biography": "Biyografi",
        "Business": "İş ve girişimcilik",
        "Children": "Çocuk kitabı",
        "Education": "Eğitim / akademik",
        "Cooking": "Yemek",
        "Comics": "Çizgi roman",
        "Fiction": "Roman / kurgu",
        "Books": "Kitap",
    }
    return mapping.get(str(category), str(category or "Kitap"))


def build_product_text(product: dict) -> str:
    """Embedding için birleşik kitap metni oluşturur."""
    parts = []
    for field in [
        "title", "description", "category", "brand", "author", "style", "usage",
        "features", "details", "main_category"
    ]:
        value = str(product.get(field, "")).strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            parts.append(value)
    return " | ".join(parts)


def truncate_text(text: str, max_chars: int = 500) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_price_tl(price_usd: float | str) -> str:
    """Amazon USD fiyatını demo amacıyla TL formatında gösterir."""
    try:
        raw = str(price_usd).replace("$", "").replace(",", ".").strip()
        if raw.lower() in {"", "none", "nan", "null"}:
            return "Fiyat belirtilmemiş"
        usd = float(raw)
        if usd <= 0:
            return "Fiyat belirtilmemiş"
        tl = usd * 32.5
        return f"₺{tl:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "Fiyat belirtilmemiş"


def clean_product_title(title: str, max_words: int = 12) -> str:
    words = str(title or "").split()
    return title if len(words) <= max_words else " ".join(words[:max_words]) + "..."



def clean_display_value(value, preferred_keys=("name", "title", "brand", "store", "author"), max_chars: int = 160) -> str:
    """Amazon metadata içindeki dict/list/string değerleri kullanıcıya temiz gösterir."""
    import ast
    import json
    import math

    if value is None:
        return ""

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "{}", "[]"}:
            return ""
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return clean_display_value(ast.literal_eval(text), preferred_keys, max_chars)
            except Exception:
                try:
                    return clean_display_value(json.loads(text), preferred_keys, max_chars)
                except Exception:
                    pass

        try:
            import re
            m = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]{2,120})['\"]", text)
            if m:
                return m.group(1).strip()[:max_chars]
        except Exception:
            pass
        return text[:max_chars]

    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass
        return str(value)[:max_chars]

    if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple, set)):
        try:
            return clean_display_value(value.tolist(), preferred_keys, max_chars)
        except Exception:
            return ""

    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                cleaned = clean_display_value(value.get(key), preferred_keys, max_chars)
                if cleaned and not cleaned.startswith("http"):
                    return cleaned[:max_chars]
        parts = []
        for k, v in value.items():
            if str(k).lower() in {"avatar", "image", "images", "url", "about", "description"}:
                continue
            cleaned = clean_display_value(v, preferred_keys, 80)
            if cleaned and not cleaned.startswith("http"):
                parts.append(f"{k}: {cleaned}")
            if len("; ".join(parts)) > max_chars:
                break
        return "; ".join(parts)[:max_chars]

    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            cleaned = clean_display_value(item, preferred_keys, 100)
            if cleaned and not cleaned.startswith("http"):
                parts.append(cleaned)
            if len(" ".join(parts)) > max_chars:
                break
        return " ".join(parts)[:max_chars]

    return str(value)[:max_chars]
