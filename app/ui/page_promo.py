# app/ui/page_promo.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from analytics.promo import calculate_promo_metrics

def show_promo_page():
    st.title("💰 Commercial Intelligence & Promo Optimization")
    st.markdown("Walmart M5 veri seti tabanlı çapraz fiyat esnekliği ve ürün yamyamlaşma analizi.")
    
    # 🟢 SAHTE / DEMO VERİ SETİ OLUŞTURMA (Walmart M5 yapısında)
    # Gerçek veri setini yüklediğinde burayı pd.read_csv ile değiştireceksin
    @st.cache_data
    def load_demo_data():
        dates = pd.date_range(start="2025-01-01", periods=180)
        np.random.seed(42)
        # Klasik Kola Verileri
        kola_price = np.random.choice([2.5, 2.0, 2.5], size=180, p=[0.7, 0.2, 0.1]) # İndirim günleri
        kola_sales = 1000 - (kola_price * 200) + np.random.normal(0, 50, 180)
        # Diyet Kola Verileri (Kardeş ürün - Yamyamlaşma adayı)
        diyet_price = np.random.choice([2.7, 2.7, 2.7], size=180) # Sabit fiyat
        diyet_sales = 600 + (kola_price * 100) + np.random.normal(0, 30, 180) # Kola ucuzlayınca diyet düşer
        
        return pd.DataFrame({
            "date": dates,
            "Kola_price": kola_price, "Kola_sales": kola_sales,
            "Diyet_Kola_price": diyet_price, "Diyet_Kola_sales": diyet_sales
        })
    
    df = load_demo_data()
    
    # --- ARAYÜZ BİLEŞENLERİ ---
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Hedef Ürün (Analiz Edilecek)", ["Kola"])
    with col2:
        competitor = st.selectbox("Etkileşimdeki Kardeş/Rakip Ürün", ["Diyet_Kola"])
        
    # Hesaplamayı Tetikle
    metrics = calculate_promo_metrics(df, target, competitor)
    
    # 📊 BÖLÜM 1: KPI KARTLARI
    st.markdown("### 📈 Promosyon Performans Metrikleri")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric(
            label="Ortalama Promosyon Lifti", 
            value=f"+%{metrics['avg_promo_lift_pct']}",
            delta="Hedef Üstü Satış"
        )
    with kpi2:
        # Pozitif çapraz esneklik yamyamlaşmayı (ikame etkisini) gösterir
        status = "⚠️ Yüksek Risk" if metrics['cross_elasticity'] > 0.3 else "🟢 Güvenli"
        st.metric(
            label="Yamyamlaşma Riski (Cross-Elasticity)", 
            value=f"{metrics['cross_elasticity']}",
            delta=status,
            delta_color="inverse"
        )
    with kpi3:
        st.metric(
            label="Kendi Fiyat Esnekliği", 
            value=f"{metrics['own_elasticity']}",
            delta="Negatif olması normaldir"
        )
        
    st.write("---")
    
    # 🎛️ BÖLÜM 2: PROMOSYON SİMÜLATÖRÜ (SENARYO ANALİZİ)
    st.markdown("### 🔮 Canlı Senaryo Simülatörü")
    st.write("Aşağıdaki slider ile hedef ürüne indirim uygulandığında sistemin vereceği stok ve talep tepkisini simüle edin.")
    
    indirim_orani = st.slider("Uygulanacak İndirim Oranı (%)", 0, 50, 20, step=5)
    
    # Simülasyon Matematiği
    eski_satis = df[target + '_sales'].mean()
    # Esneklik formülü: % Talep Değişimi = Esneklik * % Fiyat Değişimi
    # Own elasticity negatif olduğu için eksi ile çarparak talebi artırıyoruz
    talep_artisi = -metrics['own_elasticity'] * (indirim_orani / 100)
    yeni_satis = eski_satis * (1 + talep_artisi)
    
    # Yamyamlaşan ürün tahmini
    # Kola ucuzlayınca diyet kola düşer (cross elasticity pozitifse)
    diyet_kaybi = metrics['cross_elasticity'] * (indirim_orani / 100)
    eski_diyet = df[competitor + '_sales'].mean()
    yeni_diyet = eski_diyet * (1 - diyet_kaybi)
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.success(f"🎯 **{target}** Talebi: **{int(yeni_satis)} adet** (+%{round(talep_artisi*100, 1)})")
        st.caption(r"Öneri: Lojistik ekibine haber verin, emniyet stoğunu artırın.")
    with sim_col2:
        st.warning(f"⚠️ **{competitor}** Talebi: **{int(yeni_diyet)} adet** (-%{round(diyet_kaybi*100, 1)})")
        st.caption(r"Öneri: Depoda fazla stok kalmaması için tedarik hızını yavaşlatın.")