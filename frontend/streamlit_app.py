import requests
import streamlit as st

API_BASE = "http://localhost:8000"
API_TIMEOUT = 240

# Sayfa ayarı
st.set_page_config(
    page_title="Hibrit Kitap Öneri Sistemi",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown("""
<style>
  .product-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid #e0e0e0;
  }
  .product-title { font-size: 15px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
  .product-price { font-size: 18px; font-weight: 800; color: #e63946; margin-bottom: 8px; }
  .product-meta  { font-size: 12px; color: #555; margin-bottom: 4px; }
  .explanation   { font-size: 13px; color: #2d6a4f; background: #d8f3dc;
                   padding: 10px; border-radius: 8px; margin-top: 8px; }
  .signal-badge  { display: inline-block; font-size: 11px; padding: 2px 8px;
                   border-radius: 12px; background: #457b9d; color: white; margin-top: 6px; }
  .section-header { font-size: 22px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
  .sub-header    { font-size: 14px; color: #666; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Yardımcı fonksiyonlar
def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API bağlantısı kurulamadı. FastAPI sunucusunun çalıştığından emin olun.")
        return None
    except Exception as e:
        st.error(f"API Hatası: {e}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API bağlantısı kurulamadı.")
        return None
    except Exception as e:
        st.error(f"API Hatası: {e}")
        return None


def signal_label(signal: str) -> str:
    labels = {
        "semantic":      "🔍 Semantik Uyum",
        "collaborative": "👥 Kullanıcı Davranışı",
        "content":       "📄 İçerik Benzerliği",
        "popularity":    "⭐ Popüler Ürün",
        "category":      "🏷️ Kategori Uyumu",
    }
    return labels.get(signal, signal)


def render_product_card(
    rec: dict,
    show_similar: bool = True,
    user_query: str = "",
    key_prefix: str = "card",
    index: int = 0,
):
    """Tek bir ürün kartı render eder."""
    p = rec.get("product", {})
    explanation = rec.get("explanation", "")
    signal = rec.get("dominant_signal", "semantic")
    score = rec.get("hybrid_score", 0.0)

    with st.container():
        col_img, col_info = st.columns([1, 4])

        with col_img:
            img_url = p.get("image_url", "")
            if img_url and img_url.startswith("http"):
                try:
                    st.image(img_url, use_container_width=True)
                except Exception:
                    render_placeholder_image(p.get("category", ""), p.get("title", ""))
            else:
                render_placeholder_image(p.get("category", ""), p.get("title", ""))

        with col_info:
            st.markdown(f"<div class='product-title'>{p.get('title', '')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='product-price'>{p.get('price_tl', 'Fiyat belirtilmemiş')}</div>", unsafe_allow_html=True)

            meta_parts = []
            for field, label in [("brand", "Yazar/Yayıncı"), ("category", "Kategori"),
                                   ("color", "Renk"), ("material", "Materyal"),
                                   ("style", "Stil"), ("usage", "Kullanım")]:
                val = str(p.get(field, "") or "").strip()
                if val and val.lower() != "nan":
                    meta_parts.append(f"**{label}:** {val}")

            if meta_parts:
                st.markdown("  |  ".join(meta_parts[:3]))
            if len(meta_parts) > 3:
                st.markdown("  |  ".join(meta_parts[3:]))

            desc = p.get("description", "")
            if desc and desc != "nan":
                st.markdown(f"<div class='product-meta'>{desc[:180]}{'...' if len(desc) > 180 else ''}</div>",
                            unsafe_allow_html=True)

            if explanation:
                st.markdown(
                    f"<div class='explanation'>💡 <strong>Kısa gerekçe</strong><br>{explanation}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='signal-badge'>{signal_label(signal)}</div>",
                unsafe_allow_html=True,
            )

            # Qwen/LLM açıklaması yalnızca kullanıcı isterse çağrılır.
            unique_suffix = f"{key_prefix}_{index}_{p.get('product_id', '')}_{signal}"
            detail_key = f"detail_exp_{unique_suffix}"
            if st.button("💬 Ayrıntılı açıklama oluştur", key=f"btn_{detail_key}"):
                detailed = api_post("/explain", {
                    "product_id": p.get("product_id", ""),
                    "product_title": p.get("title", ""),
                    "product_description": p.get("description", ""),
                    "product_category": p.get("category", ""),
                    "user_query": user_query or "",
                    "dominant_signal": signal,
                })
                if detailed:
                    st.session_state[detail_key] = detailed.get("explanation", "")
            if detail_key in st.session_state:
                st.info(st.session_state[detail_key])

        # Benzer ürünler: otomatik API çağrısı yapma; kullanıcı isterse getir.
        if show_similar:
            pid = p.get("product_id", "")
            if pid:
                key = f"similar_{key_prefix}_{index}_{pid}"
                if st.button("🔗 Benzer kitapları göster", key=f"btn_{key}"):
                    similar_data = api_get(f"/similar/{pid}", {"k": 4})
                    if similar_data:
                        st.session_state[key] = similar_data.get("similar", [])
                if key in st.session_state:
                    sims = st.session_state.get(key, [])
                    if sims:
                        st.markdown("**Benzer kitaplar:**")
                        for sim in sims[:4]:
                            sp = sim.get("product", {})
                            st.markdown(f"• **{sp.get('title', '')}** — {sp.get('category', '')} — {sp.get('price_tl', '')}")
                    else:
                        st.info("Benzer kitap bulunamadı.")

        st.markdown("---")


def render_placeholder_image(category: str, title: str):
    """Görsel yoksa kategori rengi ile placeholder gösterir."""
    colors = {
        "Beauty": "#f8c8d4", "Shoes": "#c8e6f8", "Bags": "#f8e4c8",
        "Electronics": "#c8d4f8", "Home": "#d4f8c8", "Clothing": "#f4c8f8",
        "Jewelry": "#f8f4c8",
    }
    bg = colors.get(category, "#e8e8e8")
    emoji = {"Beauty": "💄", "Shoes": "👟", "Bags": "👜",
              "Electronics": "🎧", "Home": "🏠", "Clothing": "👕", "Jewelry": "💍"}.get(category, "📦")
    st.markdown(
        f"""<div style="background:{bg};border-radius:10px;padding:20px;
        text-align:center;min-height:100px;display:flex;align-items:center;
        justify-content:center;flex-direction:column;">
        <div style="font-size:36px">{emoji}</div>
        <div style="font-size:11px;color:#555;margin-top:6px">{category}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# Başlık
st.markdown("# 📚 Hibrit Kitap Öneri Sistemi")
st.markdown("*Okuma geçmişinize ve arama niyetinize göre kişiselleştirilmiş kitap keşfi*")
st.markdown("---")

# API sağlık kontrolü
health = api_get("/health")
if health:
    if health.get("status") == "ok":
        stats = health.get("stats", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Ürün", stats.get("product_count", "—"))
        col2.metric("Kullanıcı", stats.get("user_count", "—"))
        col3.metric("Etkileşim", stats.get("interaction_count", "—"))
    else:
        st.warning("⚠️ Sistem başlatılıyor, lütfen bekleyin...")

# Sekmeler
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Kişisel Öneriler",
    "🆕 Yeni Okuyucu",
    "🔍 Kitap Arama",
    "⚖️ Karşılaştırma",
    "📊 Sistem Bilgisi",
])

# SEKME 1: KİŞİSELLEŞTİRİLMİŞ ÖNERİ

with tab1:
    st.markdown("<div class='section-header'>🎯 Size Özel Kitap Önerileri</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Okuma geçmişinize ve isteğe bağlı arama ifadenize göre kitap önerileri</div>",
                unsafe_allow_html=True)

    users_data = api_get("/users")
    if not users_data:
        st.stop()

    user_options = {u["display_name"]: u["user_id"] for u in users_data.get("users", [])}

    col_sel, col_pref, col_k = st.columns([2, 3, 1])
    with col_sel:
        selected_name = st.selectbox("👤 Okuyucu seç", list(user_options.keys()))
    with col_pref:
        query = st.text_input("🔎 Aradığınız tür veya konu (opsiyonel)", placeholder="Örn: korku romanı, bilim kurgu, psikoloji kitabı")
    with col_k:
        k1 = st.number_input("Öneri sayısı", min_value=2, max_value=16, value=6, key="k1")

    if st.button("✨ Kitap Öner", type="primary", key="btn1"):
        user_id = user_options[selected_name]
        with st.spinner("Profil analiz ediliyor..."):
            profile = api_get(f"/users/{user_id}/profile")
        with st.spinner("Öneriler hesaplanıyor..."):
            recs_data = api_get("/recommend", {"user_id": user_id, "k": k1, "preferences": query})
        st.session_state["tab1_profile"] = profile
        st.session_state["tab1_recs_data"] = recs_data
        st.session_state["tab1_selected_name"] = selected_name
        st.session_state["tab1_query"] = query

    profile = st.session_state.get("tab1_profile")
    recs_data = st.session_state.get("tab1_recs_data")
    last_name = st.session_state.get("tab1_selected_name", selected_name)
    last_query = st.session_state.get("tab1_query", query)

    if profile:
        with st.expander(f"👤 {last_name} Okuma Profili", expanded=True):
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Toplam Etkileşim", profile.get("total_interactions", 0))
            with p_col2:
                st.metric("Ort. Fiyat İlgisi", profile.get("avg_price_interest", "—"))
            with p_col3:
                cats = profile.get("top_categories", [])
                st.markdown("**📂 En Çok İlgilenilen Kategoriler**")
                for c in cats[:3]:
                    st.markdown(f"• {c}")
            with p_col4:
                recent = profile.get("recent_interactions", [])
                st.markdown("**🕐 Son Etkileşimler**")
                for r in recent[:3]:
                    st.markdown(f"• {r}")

    if recs_data:
        recs = recs_data.get("recommendations", [])
        latency = recs_data.get("latency_ms", 0)
        st.success(f"✅ {len(recs)} öneri üretildi · ⏱️ {latency:.0f}ms")
        for idx, rec in enumerate(recs):
            render_product_card(rec, show_similar=True, user_query=last_query, key_prefix="personal", index=idx)

# SEKME 2: YENİ KULLANICI / COLD-START
with tab2:
    st.markdown("<div class='section-header'>🆕 Yeni Okuyucu İçin Öneriler</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Okuma geçmişi olmayan kullanıcılar için tercih metnine göre kitap önerisi</div>",
        unsafe_allow_html=True,
    )

    st.caption("İlk kez gelen okuyucular, yalnızca ne aradıklarını yazarak kitap keşfine başlayabilir.")

    col_n, col_p = st.columns([1, 2])
    with col_n:
        new_name = st.text_input("👤 Adınız", placeholder="Örn: Yaren D.")
    with col_p:
        preferences = st.text_input(
            "💬 Ne arıyorsunuz?",
            placeholder="Örn: bilim kurgu roman, kişisel gelişim, romantik roman..."
        )
    k2 = st.slider("Öneri sayısı", 2, 12, 6, key="k2")

    if st.button("📚 Bana Kitap Öner", type="primary", key="btn2"):
        if not preferences.strip():
            st.warning("Lütfen ne aradığınızı yazın.")
        else:
            with st.spinner("Tercihlerinize göre kitaplar aranıyor..."):
                result = api_get("/recommend/new-user", {
                    "name": new_name or "Misafir",
                    "preferences": preferences,
                    "k": k2,
                })
            st.session_state["tab2_result"] = result
            st.session_state["tab2_name"] = new_name or "Kullanıcı"
            st.session_state["tab2_preferences"] = preferences

    result = st.session_state.get("tab2_result")
    if result:
        recs = result.get("recommendations", [])
        st.success(f"✅ {st.session_state.get('tab2_name', 'Kullanıcı')} için {len(recs)} öneri üretildi")
        for idx, rec in enumerate(recs):
            render_product_card(rec, show_similar=False, user_query=st.session_state.get("tab2_preferences", ""), key_prefix="new_reader", index=idx)

# SEKME 3: SEMANTİK ARAMA
with tab3:
    st.markdown("<div class='section-header'>🔍 Kitap Arama</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Türkçe doğal dil ile kitap, tür veya konu arayın.</div>",
        unsafe_allow_html=True,
    )

    example_queries = [
        "bilim kurgu roman",
        "romantik roman",
        "korku romanı",
        "psikoloji kitabı",
        "kişisel gelişim",
        "tarih kitabı",
    ]

    st.markdown("**Örnek sorgular:**")
    eq_cols = st.columns(len(example_queries))
    for i, eq in enumerate(example_queries):
        if eq_cols[i].button(eq, key=f"eq_{i}"):
            st.session_state["search_query"] = eq

    search_q = st.text_input(
        "🔎 Aramak istediğinizi yazın",
        value=st.session_state.get("search_query", ""),
        placeholder="Örn: bilim kurgu roman, psikoloji kitabı, korku romanı...",
        key="search_input",
    )
    k3 = st.slider("Sonuç sayısı", 2, 16, 8, key="k3")

    if st.button("🔍 Ara", type="primary", key="btn3") or st.session_state.get("search_query"):
        query_to_use = search_q or st.session_state.get("search_query", "")
        if query_to_use:
            with st.spinner(f"'{query_to_use}' için kitaplar aranıyor..."):
                result = api_get("/search", {"q": query_to_use, "k": k3})
            st.session_state["tab3_result"] = result
            st.session_state["tab3_query"] = query_to_use
            st.session_state["search_query"] = ""

    result = st.session_state.get("tab3_result")
    if result:
        query_to_use = st.session_state.get("tab3_query", "")
        recs = result.get("results", [])
        latency = result.get("latency_ms", 0)
        st.success(f"✅ '{query_to_use}' için {len(recs)} sonuç bulundu · ⏱️ {latency:.0f}ms")
        for idx, rec in enumerate(recs):
            render_product_card(rec, show_similar=False, user_query=query_to_use, key_prefix="search", index=idx)

# SEKME 4: KİTAP KARŞILAŞTIRMA
with tab4:
    st.markdown("<div class='section-header'>⚖️ Kitap Karşılaştırma</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>İki veya üç kitabı temel katalog bilgileri üzerinden karşılaştırın </div>",
        unsafe_allow_html=True,
    )

    search_term = st.text_input("🔎 Kitap ara (karşılaştırmak için seçin)", placeholder="Örn: aşk romanı, bilim kurgu, psikoloji...")

    if search_term:
        products_data = api_get("/products", {"q": search_term, "limit": 30})
        if products_data:
            st.session_state["tab4_products"] = products_data.get("products", [])
            st.session_state["tab4_search"] = search_term

    product_list = st.session_state.get("tab4_products", [])
    if product_list:
        product_options = {
            f"{idx+1}. {p['title'][:70]} — {p['category']} — {p['product_id']}": p["product_id"]
            for idx, p in enumerate(product_list)
        }

        selected_products = st.multiselect(
            "Karşılaştırmak istediğiniz kitapları seçin (2-3 kitap)",
            options=list(product_options.keys()),
            max_selections=3,
            key="compare_selection",
        )
        usage_ctx = st.text_input(
            "Kullanım amacı (opsiyonel)", placeholder="Örn: hediye için, bilim kurgu okumaya başlamak için...", key="compare_usage"
        )

        if st.button("⚖️ Karşılaştır", type="primary", key="btn4"):
            if len(selected_products) < 2:
                st.warning("En az 2 kitap seçin.")
            else:
                pids = [product_options[p] for p in selected_products]
                with st.spinner("Karşılaştırma yapılıyor..."):
                    result = api_post("/compare", {
                        "product_ids": pids,
                        "usage_context": usage_ctx,
                    })
                st.session_state["tab4_compare_result"] = result

    elif search_term:
        st.info("Bu aramayla karşılaştırılacak kitap bulunamadı. Daha genel bir ifade deneyin: örn. 'roman', 'fiction', 'science fiction'.")

    result = st.session_state.get("tab4_compare_result")
    if result:
        st.markdown("### 📊 Özellik Karşılaştırması")
        import pandas as pd
        table = result.get("comparison_table", [])
        if table:
            df_compare = pd.DataFrame(table)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
        else:
            st.warning("Karşılaştırma tablosu oluşturulamadı.")

# SEKME 5: SİSTEM & DEĞERLENDİRME
with tab5:
    st.markdown("<div class='section-header'>📊 Sistem Bilgisi</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Kullanılan Bileşenler")
        model_info = api_get("/model-info")
        if model_info:
            with st.container():
                st.markdown(f"**LLM Modeli:** `{model_info['llm_modeli']['ad']}`")
                st.markdown(f"*{model_info['llm_modeli']['amaç']}*")
                st.markdown("---")
                st.markdown(f"**Embedding Modeli:** `{model_info['embedding_modeli']['ad']}`")
                st.markdown(f"*{model_info['embedding_modeli']['dil_desteği']}*")

        st.markdown("### Öneri Skoru Mantığı")
        st.markdown("Öneri listesi; okuma davranışı, kitap içeriği, doğal dil araması ve genel kullanıcı ilgisi birlikte değerlendirilerek oluşturulur. Yeni okuyucularda geçmiş olmadığı için tercih metni ve kitap açıklamaları daha ağırlıklı kullanılır.")

        st.markdown("### Yöntem Özeti")
        sys_data = api_get("/metrics/system")
        if sys_data:
            for method in sys_data.get("methods_used", []):
                st.markdown(f"• {method}")

    with col_right:
        st.markdown("### 📈 Veri Seti Özeti")
        if sys_data:
            import pandas as pd
            df_stats = pd.DataFrame([
                {"Metrik": "Veri Modu", "Değer": sys_data.get("data_mode", "")},
                {"Metrik": "Ürün Sayısı", "Değer": sys_data.get("product_count", 0)},
                {"Metrik": "Kullanıcı Sayısı", "Değer": sys_data.get("user_count", 0)},
                {"Metrik": "Etkileşim Sayısı", "Değer": sys_data.get("interaction_count", 0)},
            ])
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        st.markdown("### ⏱️ Performans")
        if st.button("Performans bilgisini getir", key="btn_latency"):
            latency = api_get("/metrics/latency")
            if latency:
                import pandas as pd
                df_lat = pd.DataFrame([
                    {"Metrik": "Ort. Öneri Gecikmesi", "Değer": f"{latency.get('avg_recommendation_latency_ms', 0):.1f} ms"},
                    {"Metrik": "Ort. Arama Gecikmesi", "Değer": f"{latency.get('avg_search_latency_ms', 0):.1f} ms"},
                    {"Metrik": "Toplam Öneri İsteği", "Değer": latency.get("recommendation_requests", 0)},
                    {"Metrik": "Toplam Arama İsteği", "Değer": latency.get("search_requests", 0)},
                ])
                st.dataframe(df_lat, use_container_width=True, hide_index=True)
        else:
            st.caption("Performans endpointi yalnızca butona basıldığında çağrılır; sayfa açılışını yavaşlatmaz.")

    # Değerlendirme metrikleri
    st.markdown("---")
    st.markdown("### 🎯 Öneri Kalite Değerlendirmesi")
    st.info(
             "Temporal holdout protokolü: Kullanıcıların en güncel 2–3 etkileşimi test için ayrılır; "
             "öneriler yalnızca önceki etkileşimler üzerinden üretilir."
    )

    col_ek, col_eu = st.columns(2)
    with col_ek:
        eval_k = st.number_input("K değeri", min_value=5, max_value=20, value=10)
    with col_eu:
        eval_users = st.number_input("Değerlendirilecek kullanıcı sayısı", min_value=5, max_value=100, value=20)

    if st.button("🧮 Değerlendirmeyi Başlat", key="btn_eval"):
        with st.spinner(f"Değerlendirme hesaplanıyor ({eval_users} kullanıcı, K={eval_k})..."):
            eval_result = api_get("/metrics/evaluation", {"k": eval_k, "max_users": eval_users})

        if eval_result:
            import pandas as pd
            df_eval = pd.DataFrame([
                {"Metrik": "Precision@K", "Değer": f"{eval_result.get('precision_at_k', 0):.4f}"},
                {"Metrik": "Recall@K",    "Değer": f"{eval_result.get('recall_at_k', 0):.4f}"},
                {"Metrik": "NDCG@K",      "Değer": f"{eval_result.get('ndcg_at_k', 0):.4f}"},
                {"Metrik": "HitRate@K",   "Değer": f"{eval_result.get('hit_rate_at_k', 0):.4f}"},
                {"Metrik": "Değerlendirilen Kullanıcı", "Değer": eval_result.get("evaluated_users", 0)},
                {"Metrik": "K",           "Değer": eval_result.get("k", eval_k)},
            ])
            st.dataframe(df_eval, use_container_width=True, hide_index=True)
