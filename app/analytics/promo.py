# app/analytics/promo.py
import pandas as pd
import numpy as np

def calculate_promo_metrics(data, target_product, competitor_product):
    """
    Seçilen iki FMCG ürünü arasındaki fiyat esnekliğini ve 
    yamyamlaşma (cannibalization) etkisini hesaplar.
    """
    # 1. Sonsuz veya kayıp değerleri temizle (Veri Temizliği)
    df = data[[target_product + '_sales', target_product + '_price', 
               competitor_product + '_sales', competitor_product + '_price']].dropna()
    
    # Sıfır değerlerin logaritmasını almamak için küçük bir düzeltme (smoothing)
    df = df[(df > 0).all(axis=1)]
    
    # 2. Log-Log dönüşümleri (Esneklik katsayıları için)
    df['log_q_target'] = np.log(df[target_product + '_sales'])
    df['log_p_target'] = np.log(df[target_product + '_price'])
    df['log_p_comp'] = np.log(df[competitor_product + '_price'])
    
    # 3. İstatistiksel Korelasyon ve Esneklik Hesabı
    # Target ürünün kendi fiyat esnekliği (Own-Price Elasticity)
    own_elasticity = df['log_q_target'].corr(df['log_p_target'])
    
    # Çapraz Fiyat Esnekliği (Cross-Price Elasticity -> Yamyamlaşma Göstergesi)
    # Rakip/kardeş ürünün fiyatı düşerken bizim satışımız düşüyor mu?
    cross_elasticity = df['log_q_target'].corr(df['log_p_comp'])
    
    # 4. Baseline Satış Tahmini (Basit hareketli ortalama ile promosyonsuz dönem)
    # Gerçek projede bunu gelişmiş zaman serisi modeline bağlayabilirsin
    df['baseline'] = df[target_product + '_sales'].rolling(window=7, min_periods=1).mean()
    df['lift'] = df[target_product + '_sales'] - df['baseline']
    
    total_lift = int(df['lift'].sum()) if df['lift'].sum() > 0 else 0
    avg_promo_lift_pct = round((df[target_product + '_sales'].mean() / df['baseline'].mean() - 1) * 100, 2)
    
    return {
        "own_elasticity": round(own_elasticity, 2),
        "cross_elasticity": round(cross_elasticity, 2),
        "total_lift": total_lift,
        "avg_promo_lift_pct": avg_promo_lift_pct,
        "processed_df": df
    }