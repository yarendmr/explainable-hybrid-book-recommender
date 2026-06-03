import pytest

from recommender.scoring import get_personalized_weights, get_cold_start_weights


class TestHybridWeights:
    def test_personalized_weights_sum_to_one(self):
        weights = get_personalized_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_cold_start_weights_sum_to_one(self):
        weights = get_cold_start_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_personalized_weights_are_in_valid_range(self):
        weights = get_personalized_weights()
        for name, value in weights.items():
            assert 0.0 <= value <= 1.0, f"{name} ağırlığı 0-1 aralığında değil."

    def test_cold_start_weights_are_in_valid_range(self):
        weights = get_cold_start_weights()
        for name, value in weights.items():
            assert 0.0 <= value <= 1.0, f"{name} ağırlığı 0-1 aralığında değil."

    def test_personalized_weights_have_expected_components(self):
        weights = get_personalized_weights()
        expected = {
            "collaborative",
            "content",
            "semantic",
            "popularity",
            "category_intent",
        }
        assert set(weights.keys()) == expected

    def test_cold_start_weights_have_expected_components(self):
        weights = get_cold_start_weights()
        expected = {
            "semantic",
            "category",
            "popularity",
        }
        assert set(weights.keys()) == expected