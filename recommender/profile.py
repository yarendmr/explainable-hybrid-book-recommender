"""
Kullanıcı Profil Analizi Modülü.
"""

import logging
import pandas as pd
from app.core.schemas import UserProfile

logger = logging.getLogger(__name__)

# Demo kullanıcı görünen adları (U001 → "Ayşe D." gibi)
DEMO_USER_NAMES: dict[str, str] = {
    "U001": "Ayşe D.",
    "U002": "Yaren S.",
    "U003": "Mehmet K.",
    "U004": "Zeynep A.",
    "U005": "Fatma Ö.",
    "U006": "Ali R.",
    "U007": "Elif T.",
    "U008": "Can B.",
    "U009": "Selin Y.",
    "U010": "Hasan M.",
}


def get_display_name(user_id: str, users_df: pd.DataFrame) -> str:
    """
    Kullanıcı ID'sini görünen ada dönüştürür.
    Demo kullanıcılar için sözlük, Amazon kullanıcıları için kısaltılmış ID.
    """
    # Önce users tablosunu kontrol et
    if not users_df.empty and "display_name" in users_df.columns:
        row = users_df[users_df["user_id"] == user_id]
        if not row.empty:
            name = str(row.iloc[0]["display_name"]).strip()
            if name and name != "nan":
                return name

    if user_id in DEMO_USER_NAMES:
        return DEMO_USER_NAMES[user_id]

    # Amazon hash ID'yi kısalt
    return f"Kullanıcı {user_id[:8]}..."


def build_user_profile(
    user_id: str,
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> UserProfile:
    """ Kullanıcının etkileşim geçmişinden profil analizi oluşturur. """
    display_name = get_display_name(user_id, users_df)

    # Kullanıcı etkileşimleri
    user_df = interactions_df[interactions_df["user_id"] == user_id].copy()

    if user_df.empty:
        return UserProfile(
            user_id=user_id,
            display_name=display_name,
            top_categories=[],
            avg_price_interest="Belirsiz",
            recent_interactions=[],
            total_interactions=0,
        )

    user_df = user_df.sort_values("timestamp", ascending=False)
    total = len(user_df)

    recent_pids = user_df["product_id"].head(5).tolist()
    recent_titles = []
    for pid in recent_pids:
        row = products_df[products_df["product_id"] == pid]
        if not row.empty:
            title = str(row.iloc[0].get("title", pid))
            words = title.split()
            recent_titles.append(" ".join(words[:6]) + ("..." if len(words) > 6 else ""))

    top_categories = _top_categories(user_df, products_df)

    avg_price_str = _avg_price_interest(user_df, products_df)

    return UserProfile(
        user_id=user_id,
        display_name=display_name,
        top_categories=top_categories,
        avg_price_interest=avg_price_str,
        recent_interactions=recent_titles,
        total_interactions=total,
    )


def _top_categories(user_df: pd.DataFrame, products_df: pd.DataFrame) -> list[str]:
    """Kullanıcının en çok etkileştiği kategorileri döner."""
    cats: dict[str, int] = {}
    for pid in user_df["product_id"]:
        row = products_df[products_df["product_id"] == pid]
        if row.empty:
            continue
        cat = str(row.iloc[0].get("category", "")).strip()
        if cat and cat != "nan":
            cats[cat] = cats.get(cat, 0) + 1

    sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in sorted_cats[:4]]


def _avg_price_interest(user_df: pd.DataFrame, products_df: pd.DataFrame) -> str:
    """Kullanıcının ilgilendiği ürünlerin ortalama fiyatını hesaplar."""
    prices = []
    for pid in user_df["product_id"]:
        row = products_df[products_df["product_id"] == pid]
        if row.empty:
            continue
        price_str = str(row.iloc[0].get("price", "")).replace("$", "").strip()
        try:
            prices.append(float(price_str))
        except ValueError:
            continue

    if not prices:
        return "Belirsiz"

    avg_usd = sum(prices) / len(prices)
    avg_tl = avg_usd * 32.5
    return f"₺{avg_tl:,.0f}".replace(",", ".")
