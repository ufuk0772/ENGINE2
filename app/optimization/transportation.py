# app/optimization/transportation.py

import numpy as np


def balance_problem(supply, demand, costs):
    """
    Ulaştırma problemini dengeler.
    supply / demand liste ya da 1-D numpy dizisi olabilir.
    Her zaman (list, list, np.ndarray) döndürür.
    """
    supply = list(supply)          # mutable kopya, numpy-safe
    demand = list(demand)
    costs  = np.array(costs, dtype=float)   # garantili ndarray + kopya

    total_supply = sum(supply)
    total_demand = sum(demand)

    if total_supply > total_demand:
        diff = total_supply - total_demand
        demand.append(diff)
        costs = np.hstack([costs, np.zeros((len(supply), 1))])

    elif total_demand > total_supply:
        diff = total_demand - total_supply
        supply.append(diff)
        costs = np.vstack([costs, np.zeros((1, len(demand)))])

    return supply, demand, costs


def least_cost_method(supply, demand, costs):
    """
    En düşük maliyetli hücre yöntemi.
    Zaten doğruydu; savunmacı kopya + erken çıkış eklendi.
    """
    supply = list(supply)
    demand = list(demand)
    costs  = np.array(costs, dtype=float)

    m, n = costs.shape
    allocation = np.zeros((m, n))

    while True:
        # Tüm arz/talep karşılandıysa çık
        if sum(supply) == 0 and sum(demand) == 0:
            break

        min_cost = np.inf
        min_i, min_j = -1, -1

        for i in range(m):
            if supply[i] == 0:
                continue
            for j in range(n):
                if demand[j] == 0:
                    continue
                if costs[i, j] < min_cost:
                    min_cost = costs[i, j]
                    min_i, min_j = i, j

        if min_i == -1:   # Atanabilecek hücre kalmadı
            break

        qty = min(supply[min_i], demand[min_j])
        allocation[min_i, min_j] += qty   # += → dejenere durumda üst üste yazılmaz

        supply[min_i] -= qty
        demand[min_j] -= qty

    return allocation, float((allocation * costs).sum())


def northwest_corner_method(supply, demand, costs):
    """
    Kuzeybatı köşesi yöntemi.
    Düzeltme: supply[i] == demand[j] == 0 dead-lock giderildi.
    """
    supply = list(supply)
    demand = list(demand)
    costs  = np.array(costs, dtype=float)

    m, n = costs.shape
    allocation = np.zeros((m, n))

    i, j = 0, 0

    while i < m and j < n:
        qty = min(supply[i], demand[j])
        allocation[i, j] = qty

        supply[i] -= qty
        demand[j] -= qty

        if supply[i] == 0 and demand[j] == 0:
            # Her iki taraf da tükendi → her ikisini de ilerlet
            i += 1
            j += 1
        elif supply[i] == 0:
            i += 1
        else:
            j += 1

    return allocation, float((allocation * costs).sum())


def vogel_approximation_method(supply, demand, costs):
    """
    Vogel yaklaşım yöntemi.
    Düzeltme: tükenmiş satır/sütunlar penalty hesabından ve seçimden dışlanıyor.
    Sonsuz döngü koruması eklendi.
    """
    supply = list(supply)
    demand = list(demand)
    costs  = np.array(costs, dtype=float)

    m, n = costs.shape
    allocation = np.zeros((m, n))

    # Hâlâ aktif indeksleri takip et
    active_rows = list(range(m))
    active_cols = list(range(n))

    while active_rows and active_cols:
        # Aktif satır penalty'leri
        row_penalties = {}
        for i in active_rows:
            available = [costs[i, j] for j in active_cols]
            if len(available) >= 2:
                s = sorted(available)
                row_penalties[i] = s[1] - s[0]
            else:
                row_penalties[i] = 0

        # Aktif sütun penalty'leri
        col_penalties = {}
        for j in active_cols:
            available = [costs[i, j] for i in active_rows]
            if len(available) >= 2:
                s = sorted(available)
                col_penalties[j] = s[1] - s[0]
            else:
                col_penalties[j] = 0

        max_row_pen = max(row_penalties.values()) if row_penalties else -1
        max_col_pen = max(col_penalties.values()) if col_penalties else -1

        if max_row_pen >= max_col_pen:
            # En yüksek penalty'li satırı seç
            i = max(row_penalties, key=lambda x: row_penalties[x])
            # O satırda aktif sütunlar içinde en ucuz hücreyi seç
            j = min(active_cols, key=lambda x: costs[i, x])
        else:
            # En yüksek penalty'li sütunu seç
            j = max(col_penalties, key=lambda x: col_penalties[x])
            # O sütunda aktif satırlar içinde en ucuz hücreyi seç
            i = min(active_rows, key=lambda x: costs[x, j])

        qty = min(supply[i], demand[j])
        allocation[i, j] += qty

        supply[i] -= qty
        demand[j] -= qty

        # Tükenen satır/sütunu aktif listeden çıkar
        if supply[i] == 0:
            active_rows.remove(i)
        if demand[j] == 0:
            active_cols.remove(j)

        # Her ikisi de aynı anda tükendiyse ikisinı de çıkar (dejenere durum)
        # Yukarıdaki iki kontrol bunu zaten hallediyor.

    return allocation, float((allocation * costs).sum())

def linear_programming_method(supply, demand, costs):
    """
    Transportation problem solved via Linear Programming.
    Uses scipy.optimize.linprog
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None, None  # fallback (kırılmaz sistem)

    supply = list(supply)
    demand = list(demand)
    costs = np.array(costs, dtype=float)

    m, n = costs.shape

    c = costs.flatten()

    # Eşitlik kısıtları (supply)
    A_eq = []
    b_eq = []

    # Supply constraints
    for i in range(m):
        row = [0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])

    # Demand constraints
    for j in range(n):
        row = [0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    bounds = [(0, None)] * (m * n)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return None, None

    allocation = res.x.reshape((m, n))
    total_cost = float(res.fun)

    return allocation, total_cost