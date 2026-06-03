"""
Birim Testleri — Değerlendirme Metrikleri & Metin İşleme
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TestPrecisionAtK:
    def test_perfect(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3) == 0.0

    def test_partial(self):
        from evaluation.metrics import precision_at_k
        result = precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
        assert abs(result - 2/3) < 1e-6

    def test_k_larger_than_list(self):
        from evaluation.metrics import precision_at_k
        result = precision_at_k(["a", "b"], {"a"}, k=5)
        # İlk 5'te 1 hit; 1/5
        assert abs(result - 0.2) < 1e-6


class TestRecallAtK:
    def test_perfect(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_zero(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_empty_relevant(self):
        from evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0


class TestNDCGAtK:
    def test_perfect_ordering(self):
        from evaluation.metrics import ndcg_at_k
        # İlgili öğe ilk sırada
        result = ndcg_at_k(["a", "b", "c"], {"a"}, k=3)
        assert abs(result - 1.0) < 1e-6

    def test_worst_ordering(self):
        from evaluation.metrics import ndcg_at_k
        # İlgili öğe son sırada
        result_last = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        result_first = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        assert result_first > result_last

    def test_zero_relevant(self):
        from evaluation.metrics import ndcg_at_k
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


class TestHitRateAtK:
    def test_hit(self):
        from evaluation.metrics import hit_rate_at_k
        assert hit_rate_at_k(["x", "a", "y"], {"a"}, k=3) == 1.0

    def test_no_hit(self):
        from evaluation.metrics import hit_rate_at_k
        assert hit_rate_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_hit_outside_k(self):
        from evaluation.metrics import hit_rate_at_k
        # "a" 4. sırada ama k=3, bu yüzden miss
        assert hit_rate_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

class TestDetectCategoryIntent:
    def test_science_fiction_turkish(self):
        from recommender.text_processing import detect_category_intent
        assert detect_category_intent("bilim kurgu roman") == "Science Fiction"

    def test_romance_turkish(self):
        from recommender.text_processing import detect_category_intent
        assert detect_category_intent("romantik roman") == "Romance"

    def test_horror_turkish(self):
        from recommender.text_processing import detect_category_intent
        assert detect_category_intent("korku romanı") == "Horror"

    def test_psychology_turkish(self):
        from recommender.text_processing import detect_category_intent
        assert detect_category_intent("psikoloji kitabı") == "Psychology"

    def test_business_turkish(self):
        from recommender.text_processing import detect_category_intent
        assert detect_category_intent("girişimcilik kitabı") == "Business"

    def test_no_intent(self):
        from recommender.text_processing import detect_category_intent
        # Belirsiz sorgu
        result = detect_category_intent("iyi bir şey arıyorum")
        # None veya herhangi bir kategori olabilir, hata fırlatmamalı
        assert result is None or isinstance(result, str)


class TestFormatPriceTL:
    def test_basic(self):
        from recommender.text_processing import format_price_tl
        result = format_price_tl(10.0)
        assert "₺" in result
        assert "325" in result.replace(".", "").replace(",", "")

    def test_zero(self):
        from recommender.text_processing import format_price_tl
        result = format_price_tl(0)
        assert "belirtilmemiş" in result.lower()

    def test_string_input(self):
        from recommender.text_processing import format_price_tl
        result = format_price_tl("$19.99")
        assert "₺" in result

    def test_invalid(self):
        from recommender.text_processing import format_price_tl
        result = format_price_tl("fiyat yok")
        assert "belirtilmemiş" in result.lower()


class TestBuildProductText:
    def test_all_fields(self):
        from recommender.text_processing import build_product_text
        product = {
            "title": "Test Kitabı",
            "description": "Test açıklama",
            "category": "Science Fiction",
            "brand": "TestYazar",
        }
        text = build_product_text(product)
        assert "Test Kitabı" in text
        assert "Science Fiction" in text
        assert "TestYazar" in text

    def test_missing_fields(self):
        from recommender.text_processing import build_product_text
        product = {"title": "Yalnızca Başlık"}
        text = build_product_text(product)
        assert "Yalnızca Başlık" in text


class TestExpandQuery:
    def test_expands_turkish(self):
        from recommender.text_processing import expand_query
        result = expand_query("bilim kurgu roman")
        assert len(result) > len("bilim kurgu roman")

    def test_no_crash_empty(self):
        from recommender.text_processing import expand_query
        result = expand_query("")
        assert isinstance(result, str)
