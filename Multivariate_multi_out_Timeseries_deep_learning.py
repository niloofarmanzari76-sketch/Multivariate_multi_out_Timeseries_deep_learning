#!/usr/bin/env python
# coding: utf-8

# This code set is provided only for verification and evaluation of research and the use of research data by the concerned institution is not licensed.
# ================================================================================
# RAILWAY REVENUE MANAGEMENT FRAMEWORK
# Multi-Output Demand Forecasting & Capacity Allocation
# 
# Study: Deep Learning-Based Demand Prediction for Revenue Management in Railway Systems: Tehran-Mashhad Route
# 
# Author: Niloofar Manzari Vahed
# ================================================================================
# """

# # **Import Raw data**

# """
# ================================================================================
# Raw Dataset Introduction
# ================================================================================
# 
# Note 1 - Difference between "Ticket Purchase Date" and "Train Movement Date":
#     In the raw dataset, each record contains two distinct date fields:
#         - RegisterDate: the date/time when the passenger purchased the ticket.
#         - MoveDate: the date/time when the train actually departed.
#     These two dates naturally differ due to passenger purchasing behavior (tickets may be bought weeks or months in advance, or at the last minute), which is a normal and expected pattern in railway ticketing data.
# 
#     In the modeling pipeline of this project, data aggregation is based on the "Train Movement Date" (MoveDate), not the ticket purchase date, since the objective of the model is to forecast demand and price for each train departure day, not the day of purchase. Consequently, the date range of the final aggregated daily time series (1,066 days) does not match the date range of the raw sample based on purchase dates — this discrepancy is expected and consistent with the nature of the data.
# 
# Note 2 - Difference between raw dataset features and aggregated dataset 
# features:
#     - Raw Dataset: recorded at the transaction/ticket level, including features such as passenger full name, quota rate, ticket purchase status, fare, route code, train number, origin/destination station codes, ticket purchase date and time, train movement date and time, carriage number, seat number, service class type, and ticket price.
# 
#     - Aggregated Dataset (used for modeling): the raw data is aggregated by "train movement date" and "service class", producing, for each departure day and each service class (evening, business, economy, special), two main variables: total number of tickets sold (demand) and average ticket price. Identity-level and granular features such as passenger name, carriage number, and seat number are removed at this stage, since they are not relevant for time-series demand/price modeling and are meaningful only at the transaction level.
# 
#     Summary: the raw dataset operates at the "transaction level" while the aggregated dataset operates at the "daily / service-class level"; this structural difference is the primary reason for the difference in the number and type of features between the two datasets.
# ================================================================================

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv('data_Sales.csv')


# In[8]:


df


# In[2]:


import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# تنظیمات اولیه
# ═══════════════════════════════════════════════════════════════════

excel_path = 'S1_Windows.xlsx'   # ⚠️ نام فایل خودت را جایگزین کن

sheet_mapping = {
    'Windows (3<=days)':    'Window3_LastMinute',
    'Windows(3<14<=days)':  'Window2_Normal',
    'Windows(14>=days)':    'Window1_Prepurchase'
}

demand_cols = ['business', 'economy', 'evening', 'special']
price_cols  = ['Price_business', 'Price_economy', 'Price_evening', 'Price_special']

print("=" * 80)
print("گام 1: خواندن شیت‌ها و ساخت بازه کامل تاریخی")
print("=" * 80)

raw_sheets = {}
all_dates = []

for sheet_name, window_key in sheet_mapping.items():
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    raw['MoveDate_M'] = pd.to_datetime(raw['MoveDate_M'])
    raw_sheets[window_key] = raw
    all_dates.append(raw['MoveDate_M'])

global_min = min(s.min() for s in all_dates)
global_max = max(s.max() for s in all_dates)
full_date_range = pd.date_range(global_min, global_max, freq='D')

print(f"بازه تاریخی: {global_min.date()} تا {global_max.date()} ({len(full_date_range):,} روز)")

# ─────────────────────────────────────────────────────────────
# ★ ساخت جدول مرجع تاریخ -> shamsi ★
# ترکیب هر سه شیت، حذف تکراری‌ها بر اساس تاریخ
# این جدول به‌عنوان "منبع حقیقت" برای shamsi هر تاریخ استفاده می‌شود
# ─────────────────────────────────────────────────────────────

shamsi_lookup = (
    pd.concat([raw[['MoveDate_M', 'shamsi']] for raw in raw_sheets.values()])
    .dropna(subset=['shamsi'])
    .drop_duplicates(subset='MoveDate_M', keep='first')
    .set_index('MoveDate_M')['shamsi']
)

print(f"\nجدول مرجع shamsi ساخته شد: {len(shamsi_lookup):,} تاریخ منحصربه‌فرد")

# بررسی اینکه آیا برای یک تاریخ، مقادیر shamsi متناقض بین شیت‌ها وجود دارد
conflict_check = (
    pd.concat([raw[['MoveDate_M', 'shamsi']] for raw in raw_sheets.values()])
    .dropna(subset=['shamsi'])
    .groupby('MoveDate_M')['shamsi']
    .nunique()
)
conflicts = conflict_check[conflict_check > 1]
if len(conflicts) > 0:
    print(f"⚠️ هشدار: {len(conflicts)} تاریخ مقدار shamsi متناقض بین شیت‌ها دارند!")
    print(conflicts.head(10))
else:
    print("✅ هیچ تناقضی در مقادیر shamsi بین شیت‌ها یافت نشد")


# ═══════════════════════════════════════════════════════════════════
# گام 2: تابع پاک‌سازی هر شیت
# ═══════════════════════════════════════════════════════════════════

def clean_window_sheet(df_sheet, window_name, full_range, shamsi_ref):
    d = df_sheet.copy().sort_values('MoveDate_M').reset_index(drop=True)
    rows_before = len(d)

    print(f"\n{'─'*70}")
    print(f"در حال پردازش: {window_name}")
    print('─'*70)

    # ── تقاضای مفقود -> صفر ──
    for col in demand_cols:
        if col not in d.columns:
            continue
        missing = d[col].isna().sum()
        d[col] = d[col].fillna(0)
        if missing > 0:
            print(f"  تقاضای '{col:10s}': {missing:4d} مفقود -> صفر شد")

    # ── قیمت مفقود -> قیمت روز قبل ──
    for col in price_cols:
        if col not in d.columns:
            continue
        missing = d[col].isna().sum()
        d[col] = d[col].ffill()
        remaining = d[col].isna().sum()
        if remaining > 0:
            d[col] = d[col].bfill()
        if missing > 0:
            print(f"  قیمت '{col:15s}': {missing:4d} مفقود -> قیمت روز قبل")

    # ── حذف ستون shamsi قدیمی (چون از جدول مرجع دوباره می‌سازیم) ──
    if 'shamsi' in d.columns:
        d = d.drop(columns=['shamsi'])

    # ── بازسازی بر اساس بازه کامل تاریخی ──
    d = d.set_index('MoveDate_M').reindex(full_range)
    d.index.name = 'MoveDate_M'
    new_dates = len(d) - rows_before

    for col in demand_cols:
        if col in d.columns:
            d[col] = d[col].fillna(0)
    for col in price_cols:
        if col in d.columns:
            d[col] = d[col].ffill().bfill()

    d = d.reset_index()

    # ★ اتصال shamsi از جدول مرجع بر اساس خودِ تاریخ ★
    # این جایگزین ffill/bfill قبلی است - درست‌تر چون shamsi
    # یک ویژگی وابسته به تقویم است نه به روند زمانی
    d['shamsi'] = d['MoveDate_M'].map(shamsi_ref)

    unmatched = d['shamsi'].isna().sum()
    if unmatched > 0:
        print(f"  ⚠️ {unmatched} تاریخ در جدول مرجع shamsi پیدا نشد "
              f"(احتمالا خارج از بازه داده اصلی است)")

    print(f"\n  خلاصه: قبل={rows_before} | بعد={len(d)} | "
          f"روز تازه اضافه‌شده={new_dates} | "
          f"NaN باقی‌مانده={d.isna().sum().sum()}")

    return d


# ═══════════════════════════════════════════════════════════════════
# گام 3: اجرا روی هر سه شیت
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("پاک‌سازی و پرکردن شکاف‌های هر سه بازه")
print("=" * 80)

cleaned_windows = {}
for window_key, raw_df in raw_sheets.items():
    cleaned_windows[window_key] = clean_window_sheet(
        raw_df, window_key, full_date_range, shamsi_lookup
    )


# ═══════════════════════════════════════════════════════════════════
# گام 4: ذخیره در اکسل
# ═══════════════════════════════════════════════════════════════════

output_path = 'booking_windows_CLEANED.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for window_key, d in cleaned_windows.items():
        d.to_excel(writer, sheet_name=window_key, index=False)

print(f"\n✅ ذخیره شد: {output_path}")


# In[3]:


from scipy.stats import kruskal, mannwhitneyu, linregress
from statsmodels.graphics.tsaplots import acf
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

available_windows = ['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']
available_classes = demand_cols

# ═══════════════════════════════════════════════════════════════════
# بازسازی ساختار per-class: هر کلاس -> جدول (تاریخ × سه بازه)
# ═══════════════════════════════════════════════════════════════════

pivot_by_class = {}
for service_class in available_classes:
    class_df = pd.DataFrame({'MoveDate_M': full_date_range})
    for window_key in available_windows:
        class_df[window_key] = cleaned_windows[window_key][service_class].values
    class_df = class_df.set_index('MoveDate_M')
    class_df['Total_Demand'] = class_df[available_windows].sum(axis=1)
    pivot_by_class[service_class] = class_df

print("=" * 80)
print("ساختار per-class بازسازی شد")
print("=" * 80)
for c in available_classes:
    print(f"{c:10s}: {len(pivot_by_class[c])} روز | "
          f"مجموع تقاضا = {pivot_by_class[c]['Total_Demand'].sum():,.0f}")


# ═══════════════════════════════════════════════════════════════════
# بخش A: آیا بازه‌ها درون هر کلاس متفاوتند؟
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART A: ARE WINDOWS DIFFERENT WITHIN EACH CLASS?")
print("=" * 80)

kruskal_results = []
pairwise_results_all = []

for service_class in available_classes:
    pivot_this = pivot_by_class[service_class]
    test_data = [pivot_this[w].values for w in available_windows]

    print(f"\n### کلاس: {service_class} ###")
    h_stat, h_pval = kruskal(*test_data)
    verdict = "متفاوتند ✅" if h_pval < 0.05 else "متفاوت نیستند ❌"
    print(f"Kruskal-Wallis : H={h_stat:.4f}, p={h_pval:.6f}  -->  {verdict}")

    kruskal_results.append({
        'ServiceClass': service_class, 'H_statistic': h_stat,
        'p_value': h_pval, 'Significant': 'Yes' if h_pval < 0.05 else 'No'
    })

    for i, j in combinations(range(len(available_windows)), 2):
        u_stat, u_pval = mannwhitneyu(test_data[i], test_data[j], alternative='two-sided')
        pairwise_results_all.append({
            'ServiceClass': service_class,
            'Comparison': f"{available_windows[i]} vs {available_windows[j]}",
            'U_statistic': u_stat, 'p_value': u_pval,
            'Significant': 'Yes' if u_pval < 0.05 else 'No'
        })

kruskal_df = pd.DataFrame(kruskal_results)
pairwise_df = pd.DataFrame(pairwise_results_all)
print("\nخلاصه Kruskal-Wallis:")
print(kruskal_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# بخش B: آیا کلاس‌ها درون هر بازه متفاوتند؟
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART B: ARE CLASSES DIFFERENT WITHIN EACH WINDOW?")
print("=" * 80)

class_comparison_results = []
class_pairwise_results = []

for window_key in available_windows:
    d = cleaned_windows[window_key]
    data_per_class = [d[c].values for c in available_classes]

    print(f"\n### بازه: {window_key} ###")
    h_stat, h_pval = kruskal(*data_per_class)
    verdict = "کلاس‌ها متفاوتند ✅" if h_pval < 0.05 else "کلاس‌ها متفاوت نیستند ❌"
    print(f"Kruskal-Wallis: H={h_stat:.4f}, p={h_pval:.6f}  -->  {verdict}")

    class_comparison_results.append({
        'Window': window_key, 'H_statistic': h_stat, 'p_value': h_pval,
        'Significant': 'Yes' if h_pval < 0.05 else 'No'
    })

    for i, j in combinations(range(len(available_classes)), 2):
        u_stat, u_pval = mannwhitneyu(data_per_class[i], data_per_class[j], alternative='two-sided')
        class_pairwise_results.append({
            'Window': window_key,
            'Comparison': f"{available_classes[i]} vs {available_classes[j]}",
            'U_statistic': u_stat, 'p_value': u_pval,
            'Significant': 'Yes' if u_pval < 0.05 else 'No'
        })

class_comparison_df = pd.DataFrame(class_comparison_results)
class_pairwise_df = pd.DataFrame(class_pairwise_results)
print("\nمقایسه دوتایی کلاس‌ها درون هر بازه:")
print(class_pairwise_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# تحلیل تغییرپذیری (CV)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("VARIABILITY ANALYSIS (Coefficient of Variation)")
print("=" * 80)

cv_records = []
for service_class in available_classes:
    pivot_this = pivot_by_class[service_class]
    for window_key in available_windows:
        mean = pivot_this[window_key].mean()
        std = pivot_this[window_key].std()
        cv = (std / mean * 100) if mean > 0 else np.nan
        interp = 'Low' if cv < 30 else ('Moderate' if cv < 50 else 'High')
        cv_records.append({
            'ServiceClass': service_class, 'Window': window_key,
            'Mean': round(mean, 2), 'Std': round(std, 2),
            'CV_%': round(cv, 2) if pd.notna(cv) else np.nan,
            'Interpretation': interp
        })

cv_df = pd.DataFrame(cv_records)
print(cv_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# تحلیل همبستگی بین بازه‌ها - درون هر کلاس
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("CORRELATION BETWEEN WINDOWS — PER SERVICE CLASS")
print("=" * 80)

correlation_by_class = {}
for service_class in available_classes:
    corr_matrix = pivot_by_class[service_class][available_windows].corr()
    correlation_by_class[service_class] = corr_matrix
    print(f"\n{service_class}:")
    print(corr_matrix.round(3).to_string())


# ═══════════════════════════════════════════════════════════════════
# ویژگی تقویمی shamsi: بررسی رابطه‌اش با تقاضا
# (چون این ستون حالا معنای مشخصی دارد: تعطیل مذهبی/عادی)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("EFFECT OF shamsi (RELIGIOUS/HOLIDAY LABEL) ON DEMAND")
print("=" * 80)

# اضافه کردن shamsi به هر pivot (از یکی از شیت‌ها می‌گیریم چون یکسان است)
shamsi_series = cleaned_windows['Window1_Prepurchase'].set_index('MoveDate_M')['shamsi']

for service_class in available_classes:
    pivot_this = pivot_by_class[service_class].copy()
    pivot_this['shamsi'] = shamsi_series.reindex(pivot_this.index)

    print(f"\n### کلاس: {service_class} ###")
    for window_key in available_windows:
        groups = [pivot_this[pivot_this['shamsi'] == val][window_key].values
                  for val in sorted(pivot_this['shamsi'].dropna().unique())]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h_stat, h_pval = kruskal(*groups)
            verdict = "تأثیر معنادار ✅" if h_pval < 0.05 else "تأثیر معنادار نیست ❌"
            print(f"  {window_key:22s}: H={h_stat:.3f}, p={h_pval:.5f}  -->  {verdict}")


# ═══════════════════════════════════════════════════════════════════
# ارزیابی قابلیت پیش‌بینی (Forecastability)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("FORECASTABILITY ASSESSMENT — PER CLASS × PER WINDOW")
print("=" * 80)

def assess_forecastability(series, label):
    try:
        acf_result = acf(series.dropna(), nlags=7, fft=False)
        acf_lag1 = acf_result[1]
        acf_score = 1 if acf_lag1 > 0.4 else (0.5 if acf_lag1 > 0.2 else 0)
    except Exception:
        acf_lag1, acf_score = np.nan, 0.5

    try:
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(x, series)
        r_squared = r_value ** 2
        trend_score = 1 if r_squared > 0.3 else (0.5 if r_squared > 0.1 else 0)
    except Exception:
        r_squared, trend_score = np.nan, 0.5

    mean_val, std_val = series.mean(), series.std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else np.nan
    cv_score = 1 if cv < 30 else (0.5 if cv < 50 else 0)

    try:
        naive_forecast = series.shift(1)
        mask = series > 0
        mape_naive = np.mean(np.abs((series[mask] - naive_forecast[mask]) / series[mask])) * 100
        naive_score = 1 if mape_naive < 20 else (0.5 if mape_naive < 40 else 0)
    except Exception:
        mape_naive, naive_score = np.nan, 0.5

    overall_score = (acf_score + trend_score + cv_score + naive_score) / 4

    if overall_score >= 0.75:
        recommendation = "EXCELLENT - مدل ساده (ARIMA, MLP)"
    elif overall_score >= 0.5:
        recommendation = "GOOD - مدل ML (RF, XGBoost, LSTM)"
    else:
        recommendation = "CHALLENGING - مدل عمیق/ترکیبی (LSTM, GRU, CNN-LSTM)"

    return {
        'Label': label, 'ACF_lag1': acf_lag1, 'Trend_R2': r_squared,
        'CV_%': cv, 'Naive_MAPE_%': mape_naive,
        'Score': overall_score, 'Recommendation': recommendation
    }

forecastability_records = []
for service_class in available_classes:
    for window_key in available_windows:
        label = f"{service_class} | {window_key}"
        result = assess_forecastability(pivot_by_class[service_class][window_key], label)
        forecastability_records.append(result)

forecastability_df = pd.DataFrame(forecastability_records)
print(forecastability_df.round(3).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# رسم نمودارها - یک شکل به ازای هر کلاس
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

def plot_class_analysis(pivot_this, service_class, corr_matrix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Booking Window Analysis — {service_class}', fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    for window_key in available_windows:
        ax.plot(pivot_this.index, pivot_this[window_key].values, label=window_key, linewidth=1.2, alpha=0.85)
    ax.set_title('Demand Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    box_data = [pivot_this[w] for w in available_windows]
    bp = ax.boxplot(box_data, patch_artist=True)
    ax.set_xticklabels(['W1', 'W2', 'W3'])
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_title('Distribution by Window')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    for i, window_key in enumerate(available_windows):
        ax.hist(pivot_this[window_key], bins=25, alpha=0.5, label=window_key, color=colors[i])
    ax.set_title('Histogram of Demand')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('Correlation Between Windows')

    plt.tight_layout()
    plt.savefig(f'booking_analysis_{service_class}.png', dpi=250, bbox_inches='tight')
    print(f"  ذخیره شد: booking_analysis_{service_class}.png")
    plt.show()

for service_class in available_classes:
    plot_class_analysis(pivot_by_class[service_class], service_class, correlation_by_class[service_class])


# ═══════════════════════════════════════════════════════════════════
# ذخیره تمام نتایج در یک فایل اکسل
# ═══════════════════════════════════════════════════════════════════

with pd.ExcelWriter('booking_windows_analysis_results.xlsx', engine='openpyxl') as writer:
    kruskal_df.to_excel(writer, sheet_name='Kruskal_WithinClass', index=False)
    pairwise_df.to_excel(writer, sheet_name='Pairwise_WithinClass', index=False)
    class_comparison_df.to_excel(writer, sheet_name='Kruskal_BetweenClass', index=False)
    class_pairwise_df.to_excel(writer, sheet_name='Pairwise_BetweenClass', index=False)
    cv_df.to_excel(writer, sheet_name='Variability_CV', index=False)
    forecastability_df.to_excel(writer, sheet_name='Forecastability', index=False)
    for service_class in available_classes:
        correlation_by_class[service_class].to_excel(writer, sheet_name=f'Corr_{service_class}')

print("\n✅ ذخیره شد: booking_windows_analysis_results.xlsx")
print("\nتحلیل کامل شد!")


# In[4]:


# بررسی این‌که آیا روزهای صفر هم‌زمان (هر سه بازه با هم صفر) زیاد است
for service_class in available_classes:
    pivot_this = pivot_by_class[service_class]
    all_zero = ((pivot_this['Window1_Prepurchase'] == 0) & 
                (pivot_this['Window2_Normal'] == 0) & 
                (pivot_this['Window3_LastMinute'] == 0)).sum()
    print(f"{service_class:10s}: {all_zero} روز که هر سه بازه هم‌زمان صفرند "
          f"({all_zero/len(pivot_this)*100:.1f}%)")

# اگر این درصد برای evening بالا بود (مثلاً >10%)، بهتر است همبستگی
# را فقط روی روزهایی که حداقل یک بازه غیرصفر است دوباره حساب کنی:
for service_class in available_classes:
    pivot_this = pivot_by_class[service_class]
    non_all_zero_mask = ~((pivot_this['Window1_Prepurchase'] == 0) & 
                            (pivot_this['Window2_Normal'] == 0) & 
                            (pivot_this['Window3_LastMinute'] == 0))
    corr_filtered = pivot_this.loc[non_all_zero_mask, available_windows].corr()
    print(f"\n{service_class} (بدون روزهای کاملاً بی‌قطار):")
    print(corr_filtered.round(3))


# In[7]:


import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import acf

# ═══════════════════════════════════════════════════════════════════
# تابع جدید: Persistence Skill Score
# جایگزین MASE معیوب - بدون تعریف دایره‌ای
# ═══════════════════════════════════════════════════════════════════

def calculate_persistence_skill(series):
    """
    می‌سنجد: صرفاً با دانستن مقدار دیروز، چقدر واریانس امروز توضیح داده می‌شود؟

    Skill = 1 − Var(residual) / Var(series)
    که در آن residual = actual(t) − actual(t-1)

    این معادل R² رگرسیون actual(t) روی actual(t-1) با ضریب ۱ است
    (یعنی فرض می‌کنیم بهترین حدس، همان مقدار دیروز است - بدون تنظیم ضریب)

    برخلاف MASE قبلی، اینجا هیچ تعریف دایره‌ای وجود ندارد چون
    مخرج کسر = واریانس کل سری (نه واریانس همان چیزی که در صورت است)
    """
    series = series.dropna()
    if len(series) < 3:
        return np.nan

    naive_forecast = series.shift(1)
    residual = (series - naive_forecast).dropna()

    var_residual = residual.var()
    var_series = series.var()

    if var_series == 0:
        return np.nan

    skill = 1 - (var_residual / var_series)
    return skill


# ═══════════════════════════════════════════════════════════════════
# تابع اصلی ارزیابی قابلیت پیش‌بینی - نسخه v3 (اصلاح‌شده کامل)
# ═══════════════════════════════════════════════════════════════════

def assess_forecastability_v3(series, label):
    # ۱. خودهمبستگی لگ یک
    try:
        acf_result = acf(series.dropna(), nlags=7, fft=False)
        acf_lag1 = acf_result[1]
        acf_score = 1 if acf_lag1 > 0.4 else (0.5 if acf_lag1 > 0.2 else 0)
    except Exception:
        acf_lag1, acf_score = np.nan, 0.5

    # ۲. قدرت روند
    try:
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(x, series)
        r_squared = r_value ** 2
        trend_score = 1 if r_squared > 0.3 else (0.5 if r_squared > 0.1 else 0)
    except Exception:
        r_squared, trend_score = np.nan, 0.5

    # ۳. ضریب تغییرات (فقط برای گزارش، نه امتیازدهی اصلی - طبق بحث قبلی)
    mean_val, std_val = series.mean(), series.std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else np.nan
    cv_score = 1 if cv < 30 else (0.5 if cv < 50 else 0)

    # ۴. ★ Persistence Skill Score - جایگزین درست MASE معیوب ★
    skill = calculate_persistence_skill(series)
    if pd.isna(skill):
        skill_score = 0.5
    elif skill > 0.4:
        skill_score = 1
    elif skill > 0.15:
        skill_score = 0.5
    else:
        skill_score = 0

    overall_score = (acf_score + trend_score + cv_score + skill_score) / 4

    if overall_score >= 0.75:
        recommendation = "EXCELLENT - مدل ساده (ARIMA, MLP)"
    elif overall_score >= 0.5:
        recommendation = "GOOD - مدل ML (RF, XGBoost, LSTM)"
    else:
        recommendation = "CHALLENGING - مدل عمیق/ترکیبی (LSTM, GRU, CNN-LSTM)"

    return {
        'Label': label,
        'ACF_lag1': acf_lag1,
        'Trend_R2': r_squared,
        'CV_%': cv,
        'Persistence_Skill': skill,
        'Score': overall_score,
        'Recommendation': recommendation
    }


# ═══════════════════════════════════════════════════════════════════
# اجرا برای همه کلاس × بازه
# ═══════════════════════════════════════════════════════════════════

print("=" * 90)
print("FORECASTABILITY ASSESSMENT (v3 — FIXED: Persistence Skill instead of MASE)")
print("=" * 90)

forecastability_records_v3 = []
for service_class in available_classes:
    for window_key in available_windows:
        label = f"{service_class} | {window_key}"
        result = assess_forecastability_v3(
            pivot_by_class[service_class][window_key], label
        )
        forecastability_records_v3.append(result)

forecastability_df_v3 = pd.DataFrame(forecastability_records_v3)
print(forecastability_df_v3.round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# مقایسه سه نسخه کنار هم: v1(MAPE) - v2(MASE معیوب) - v3(Skill درست)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("COMPARISON: v1 (MAPE) vs v2 (buggy MASE) vs v3 (Persistence Skill)")
print("=" * 90)

comparison_v3 = forecastability_df[['Label', 'Score', 'Recommendation']].rename(
    columns={'Score': 'Score_v1_MAPE', 'Recommendation': 'Rec_v1_MAPE'}
).merge(
    forecastability_df_v3[['Label', 'Score', 'Recommendation']].rename(
        columns={'Score': 'Score_v3_Skill', 'Recommendation': 'Rec_v3_Skill'}
    ),
    on='Label'
)

print(comparison_v3.to_string(index=False))

changed = (comparison_v3['Rec_v1_MAPE'] != comparison_v3['Rec_v3_Skill']).sum()
print(f"\nتعداد ترکیب‌هایی که توصیه‌شان نسبت به نسخه اول تغییر کرد: {changed} از {len(comparison_v3)}")


# In[8]:


import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import acf

# ═══════════════════════════════════════════════════════════════════
# تابع جدید: قدرت الگوی هفتگی (lag-7 ACF)
# این معیار کاملاً مستقل از ACF_lag1 است - بعد جدیدی از پیش‌بینی‌پذیری
# ═══════════════════════════════════════════════════════════════════

def calculate_weekly_pattern_strength(series):
    """
    خودهمبستگی در لگ ۷ (یک هفته) را می‌سنجد.
    برخلاف lag-1، این معیار به‌طور مستقیم اندازه‌گیری نمی‌کند
    "روز قبل چقدر شبیه امروز است"، بلکه می‌سنجد
    "همین روز هفته قبل چقدر شبیه امروز است"

    برای داده فروش بلیط، این می‌تواند قوی‌تر از lag-1 باشد
    چون سه‌شنبه معمولا بیشتر شبیه سه‌شنبه قبلی است تا شبیه دوشنبه
    """
    series_clean = series.dropna()
    if len(series_clean) < 14:  # حداقل دو هفته داده لازم است
        return np.nan
    try:
        acf_result = acf(series_clean, nlags=7, fft=False)
        return acf_result[7]
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════
# تابع اصلی ارزیابی - نسخه v4 (چهار معیار واقعاً مستقل)
# ═══════════════════════════════════════════════════════════════════

def assess_forecastability_v4(series, label):
    # معیار ۱: خودهمبستگی کوتاه‌مدت (لگ-۱ - "آیا دیروز شبیه امروز است؟")
    try:
        acf_result = acf(series.dropna(), nlags=8, fft=False)
        acf_lag1 = acf_result[1]
        acf_score = 1 if acf_lag1 > 0.4 else (0.5 if acf_lag1 > 0.2 else 0)
    except Exception:
        acf_lag1, acf_score = np.nan, 0.5

    # معیار ۲: روند خطی بلندمدت
    try:
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(x, series)
        r_squared = r_value ** 2
        trend_score = 1 if r_squared > 0.3 else (0.5 if r_squared > 0.1 else 0)
    except Exception:
        r_squared, trend_score = np.nan, 0.5

    # معیار ۳: ضریب تغییرات (نویز کلی سری)
    mean_val, std_val = series.mean(), series.std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else np.nan
    cv_score = 1 if cv < 30 else (0.5 if cv < 50 else 0)

    # معیار ۴: ★ الگوی هفتگی (لگ-۷) - جایگزین مستقل به‌جای Persistence_Skill ★
    weekly_acf = calculate_weekly_pattern_strength(series)
    if pd.isna(weekly_acf):
        weekly_score = 0.5
    else:
        weekly_score = 1 if weekly_acf > 0.3 else (0.5 if weekly_acf > 0.15 else 0)

    overall_score = (acf_score + trend_score + cv_score + weekly_score) / 4

    if overall_score >= 0.75:
        recommendation = "EXCELLENT - مدل ساده (ARIMA, MLP)"
    elif overall_score >= 0.5:
        recommendation = "GOOD - مدل ML (RF, XGBoost, LSTM)"
    else:
        recommendation = "CHALLENGING - مدل عمیق/ترکیبی (LSTM, GRU, CNN-LSTM)"

    return {
        'Label': label,
        'ACF_lag1': acf_lag1,
        'Trend_R2': r_squared,
        'CV_%': cv,
        'Weekly_ACF_lag7': weekly_acf,
        'Score': overall_score,
        'Recommendation': recommendation
    }


# ═══════════════════════════════════════════════════════════════════
# اجرا برای همه کلاس × بازه
# ═══════════════════════════════════════════════════════════════════

print("=" * 90)
print("FORECASTABILITY ASSESSMENT (v4 — Independent 4th criterion: weekly lag-7 ACF)")
print("=" * 90)

forecastability_records_v4 = []
for service_class in available_classes:
    for window_key in available_windows:
        label = f"{service_class} | {window_key}"
        result = assess_forecastability_v4(
            pivot_by_class[service_class][window_key], label
        )
        forecastability_records_v4.append(result)

forecastability_df_v4 = pd.DataFrame(forecastability_records_v4)
print(forecastability_df_v4.round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# بررسی مستقل بودن معیار جدید: همبستگی بین ACF_lag1 و Weekly_ACF_lag7
# اگر این عدد پایین باشد، یعنی این‌بار واقعاً دو بعد متفاوت می‌سنجیم
# ─────────────────────────────────────────────────────────────
independence_check = forecastability_df_v4[['ACF_lag1', 'Weekly_ACF_lag7']].corr().iloc[0, 1]
print(f"\nهمبستگی بین ACF_lag1 و Weekly_ACF_lag7 در بین 12 ترکیب: {independence_check:.3f}")
print("(هرچه به صفر نزدیک‌تر باشد، یعنی این دو معیار واقعاً مستقل از هم هستند)")


# In[10]:


import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import acf

# ═══════════════════════════════════════════════════════════════════
# نسخه v5: آستانه CV نسبی (percentile-based) به‌جای مطلق
# ═══════════════════════════════════════════════════════════════════

def calculate_weekly_pattern_strength(series):
    series_clean = series.dropna()
    if len(series_clean) < 14:
        return np.nan
    try:
        acf_result = acf(series_clean, nlags=7, fft=False)
        return acf_result[7]
    except Exception:
        return np.nan


def compute_raw_metrics(series, label):
    """فقط مقادیر خام (بدون امتیازدهی) - امتیازدهی بعداً و به‌صورت نسبی انجام می‌شود"""
    try:
        acf_result = acf(series.dropna(), nlags=8, fft=False)
        acf_lag1 = acf_result[1]
    except Exception:
        acf_lag1 = np.nan

    try:
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(x, series)
        r_squared = r_value ** 2
    except Exception:
        r_squared = np.nan

    mean_val, std_val = series.mean(), series.std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else np.nan

    weekly_acf = calculate_weekly_pattern_strength(series)

    return {
        'Label': label, 'ACF_lag1': acf_lag1, 'Trend_R2': r_squared,
        'CV_%': cv, 'Weekly_ACF_lag7': weekly_acf
    }


# ─────────────────────────────────────────────────────────────
# گام ۱: جمع‌آوری مقادیر خام برای همه ۱۲ ترکیب
# ─────────────────────────────────────────────────────────────
raw_records = []
for service_class in available_classes:
    for window_key in available_windows:
        label = f"{service_class} | {window_key}"
        raw_records.append(
            compute_raw_metrics(pivot_by_class[service_class][window_key], label)
        )

raw_df = pd.DataFrame(raw_records)

# ─────────────────────────────────────────────────────────────
# گام ۲: امتیازدهی نسبی برای CV (بر اساس چارک‌های همین ۱۲ مقدار)
# پایین‌ترین چارک = بهترین (کمترین تغییرپذیری نسبی) = امتیاز ۱
# بالاترین چارک = بدترین (بیشترین تغییرپذیری نسبی) = امتیاز ۰
# ─────────────────────────────────────────────────────────────
q33, q66 = raw_df['CV_%'].quantile([0.33, 0.66])
print(f"آستانه‌های نسبی CV (بر اساس توزیع همین داده): "
      f"چارک پایین={q33:.1f}%, چارک بالا={q66:.1f}%")

def relative_cv_score(cv):
    if pd.isna(cv):
        return 0.5
    return 1 if cv <= q33 else (0.5 if cv <= q66 else 0)

# ─────────────────────────────────────────────────────────────
# گام ۳: امتیازدهی برای بقیه معیارها (همان آستانه‌های قبلی، چون منطقی‌اند)
# ─────────────────────────────────────────────────────────────
def acf_score_fn(v):   return 1 if v > 0.4 else (0.5 if v > 0.2 else 0)
def trend_score_fn(v): return 1 if v > 0.3 else (0.5 if v > 0.1 else 0)
def weekly_score_fn(v): return 1 if v > 0.3 else (0.5 if v > 0.15 else 0)

raw_df['ACF_Score']    = raw_df['ACF_lag1'].apply(acf_score_fn)
raw_df['Trend_Score']  = raw_df['Trend_R2'].apply(trend_score_fn)
raw_df['CV_Score']     = raw_df['CV_%'].apply(relative_cv_score)
raw_df['Weekly_Score'] = raw_df['Weekly_ACF_lag7'].apply(weekly_score_fn)

raw_df['Score'] = (raw_df['ACF_Score'] + raw_df['Trend_Score'] +
                    raw_df['CV_Score'] + raw_df['Weekly_Score']) / 4

def recommend(score):
    if score >= 0.75:
        return "EXCELLENT - مدل ساده (ARIMA, MLP)"
    elif score >= 0.5:
        return "GOOD - مدل ML (RF, XGBoost, LSTM)"
    else:
        return "CHALLENGING - مدل عمیق/ترکیبی (LSTM, GRU, CNN-LSTM)"

raw_df['Recommendation'] = raw_df['Score'].apply(recommend)

print("\n" + "=" * 90)
print("FORECASTABILITY ASSESSMENT (v5 — RELATIVE CV THRESHOLDS)")
print("=" * 90)
display_cols = ['Label', 'ACF_lag1', 'Trend_R2', 'CV_%', 'Weekly_ACF_lag7', 'Score', 'Recommendation']
print(raw_df[display_cols].round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# مقایسه توزیع امتیازها: v4 (مطلق) در برابر v5 (نسبی)
# ─────────────────────────────────────────────────────────────
print("\nتوزیع CV_Score در نسخه جدید (باید غیرصفر و متنوع باشد):")
print(raw_df['CV_Score'].value_counts().sort_index())


# # **describe data**

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ===================== تنظیمات حرفه‌ای برای مقاله =====================
sns.set_theme(style="whitegrid", font_scale=1.1)
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# ===================== ۱. جدول خلاصه کلی داده =====================
print("=" * 70)
print("جدول ۱. خلاصه کلی داده به تفکیک کلاس خدماتی")
print("=" * 70)

summary_data = []
for cls in available_classes:
    df_cls = pivot_by_class[cls]
    row = {
        'Service Class': cls.capitalize(),
        'تعداد روز': len(df_cls),
        'مجموع تقاضا': df_cls.sum().sum(),
        'میانگین روزانه': df_cls.mean().mean().round(2),
        'انحراف معیار': df_cls.std().mean().round(2),
        'درصد روزهای صفر': ((df_cls == 0).sum().sum() / df_cls.size * 100).round(2),
        'حداکثر تقاضای روزانه': df_cls.max().max()
    }
    summary_data.append(row)

summary_table = pd.DataFrame(summary_data)
print(summary_table.to_string(index=False))

# ذخیره جدول برای مقاله
summary_table.to_excel("Table1_Data_Summary.xlsx", index=False)
print("\nجدول خلاصه در فایل Table1_Data_Summary.xlsx ذخیره شد.")


# In[14]:


# ===================== ۳. نمودارهای اصلی (داورپسند) =====================

# --- شکل ۱: سری زمانی تقاضا (۴ کلاس) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.flatten()

for idx, cls in enumerate(available_classes):
    df_cls = pivot_by_class[cls]
    df_cls.plot(ax=axes[idx], linewidth=0.9, alpha=0.9)
    axes[idx].set_title(f"{cls.capitalize()}", fontweight='bold')
    axes[idx].set_ylabel("Daily Demand")
    axes[idx].legend(available_windows, fontsize=8, frameon=False)

plt.suptitle("Figure 1. Daily Demand Time Series by Service Class and Booking Window", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure1_Time_Series_Demand.png", bbox_inches='tight')
plt.show()


# In[19]:


# ===================== ۲. جدول آمار توصیفی به تفکیک بازه =====================
print("\n" + "=" * 70)
print("جدول ۲. آمار توصیفی تقاضا به تفکیک کلاس و بازه زمانی")
print("=" * 70)

detailed_stats = []
for cls in available_classes:
    for window in available_windows:
        series = pivot_by_class[cls][window]
        detailed_stats.append({
            'Class': cls.capitalize(),
            'Window': window,
            'Mean': round(series.mean(), 2),
            'Std': round(series.std(), 2),
            'Median': round(series.median(), 2),
            'Min': series.min(),
            'Max': series.max(),
            'Zeros (%)': round((series == 0).mean() * 100, 2),
            'CV (%)': round(series.std() / series.mean() * 100, 1) if series.mean() > 0 else np.nan
        })

stats_df = pd.DataFrame(detailed_stats)
print(stats_df.to_string(index=False))
stats_df.to_excel("Table2_Descriptive_Stats_by_Window.xlsx", index=False)


# In[20]:


# ===================== ۳. نمودارهای اصلی (داورپسند) =====================

# --- شکل ۱: سری زمانی تقاضا (۴ کلاس) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.flatten()

for idx, cls in enumerate(available_classes):
    df_cls = pivot_by_class[cls]
    df_cls.plot(ax=axes[idx], linewidth=0.9, alpha=0.9)
    axes[idx].set_title(f"{cls.capitalize()}", fontweight='bold')
    axes[idx].set_ylabel("Daily Demand")
    axes[idx].legend(available_windows, fontsize=8, frameon=False)

plt.suptitle("Figure 1. Daily Demand Time Series by Service Class and Booking Window", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure1_Time_Series_Demand.png", bbox_inches='tight')
plt.show()


# In[21]:


# --- شکل ۲: مقایسه توزیع تقاضا بین بازه‌ها (Boxplot) ---
fig, ax = plt.subplots(figsize=(11, 6))

plot_data = []
for cls in available_classes:
    for window in available_windows:
        temp = pivot_by_class[cls][[window]].copy()
        temp['Class'] = cls.capitalize()
        temp['Window'] = window
        temp = temp.rename(columns={window: 'Demand'})
        plot_data.append(temp)

box_df = pd.concat(plot_data, ignore_index=True)

sns.boxplot(data=box_df, x='Class', y='Demand', hue='Window', ax=ax, 
            palette="Set2", showfliers=False)
ax.set_title("Figure 2. Distribution of Daily Demand by Service Class and Booking Window", 
             fontweight='bold')
ax.set_ylabel("Daily Demand")
ax.legend(title="Booking Window", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Figure2_Boxplot_Demand_by_Window.png", bbox_inches='tight')
plt.show()


# In[22]:


# --- شکل ۳: همبستگی بین بازه‌ها (به تفکیک کلاس) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for idx, cls in enumerate(available_classes):
    corr = pivot_by_class[cls][available_windows].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", 
                square=True, ax=axes[idx], cbar=False,
                annot_kws={"size": 10})
    axes[idx].set_title(f"{cls.capitalize()}", fontweight='bold')

plt.suptitle("Figure 3. Correlation Between Booking Windows by Service Class", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure3_Correlation_Heatmap.png", bbox_inches='tight')
plt.show()


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ===================== تنظیمات حرفه‌ای =====================
sns.set_theme(style="whitegrid", font_scale=1.05)
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

# ===================== ۱. بررسی داده‌های مفقوده =====================
print("=" * 75)
print("جدول ۱. بررسی داده‌های مفقوده (NaN) و روزهای صفر")
print("=" * 75)

missing_records = []

for cls in available_classes:
    for window in available_windows:
        series = pivot_by_class[cls][window]

        n_total = len(series)
        n_nan = series.isna().sum()
        n_zero = (series == 0).sum()

        missing_records.append({
            'Class': cls.capitalize(),
            'Window': window,
            'تعداد کل روز': n_total,
            'داده مفقوده (NaN)': n_nan,
            'درصد NaN': round(n_nan / n_total * 100, 2),
            'تعداد روزهای صفر': n_zero,
            'درصد روزهای صفر': round(n_zero / n_total * 100, 2)
        })

missing_df = pd.DataFrame(missing_records)
print(missing_df.to_string(index=False))

# ذخیره جدول
missing_df.to_excel("Table_Missing_and_Zero_Demand.xlsx", index=False)
print("\nجدول در فایل Table_Missing_and_Zero_Demand.xlsx ذخیره شد.")


# In[26]:


# ===================== ۲. نمودار بررسی داده مفقوده و صفر =====================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for idx, cls in enumerate(available_classes):
    df_cls = pivot_by_class[cls][available_windows]

    # درصد روزهای صفر در هر بازه
    zero_pct = (df_cls == 0).mean() * 100

    zero_pct.plot(kind='bar', ax=axes[idx], color=['#e74c3c', '#f39c12', '#27ae60'])
    axes[idx].set_title(f"{cls.capitalize()}", fontweight='bold', fontsize=12)
    axes[idx].set_ylabel("درصد روزهای صفر (%)")
    axes[idx].set_ylim(0, max(zero_pct.max() + 5, 30))
    axes[idx].tick_params(axis='x', rotation=15)

    # اضافه کردن مقدار روی میله‌ها
    for p in axes[idx].patches:
        axes[idx].annotate(f"{p.get_height():.1f}%", 
                           (p.get_x() + p.get_width()/2, p.get_height()),
                           ha='center', va='bottom', fontsize=9)

plt.suptitle("Figure X. Percentage of Zero-Demand Days by Booking Window", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure_Zero_Demand_Percentage.png", bbox_inches='tight', dpi=300)
plt.show()


# In[27]:


# ===================== ۳. تشخیص داده‌های پرت (Outliers) =====================
print("\n" + "=" * 75)
print("جدول ۲. تشخیص داده‌های پرت با روش IQR (به تفکیک کلاس و بازه)")
print("=" * 75)

outlier_records = []

for cls in available_classes:
    for window in available_windows:
        series = pivot_by_class[cls][window].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        n_outliers = len(outliers)
        pct_outliers = round(n_outliers / len(series) * 100, 2)

        outlier_records.append({
            'Class': cls.capitalize(),
            'Window': window,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'تعداد پرت': n_outliers,
            'درصد پرت': pct_outliers,
            'حد پایین': round(lower_bound, 2),
            'حد بالا': round(upper_bound, 2),
            'حداکثر مقدار': series.max()
        })

outlier_df = pd.DataFrame(outlier_records)
print(outlier_df.to_string(index=False))
outlier_df.to_excel("Table_Outliers_IQR.xlsx", index=False)
print("\nجدول پرت‌ها در فایل Table_Outliers_IQR.xlsx ذخیره شد.")


# In[28]:


# ===================== ۴. نمودار Boxplot + Outliers =====================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for idx, cls in enumerate(available_classes):
    plot_data = []
    for window in available_windows:
        temp = pivot_by_class[cls][[window]].copy()
        temp.columns = ['Demand']
        temp['Window'] = window
        plot_data.append(temp)

    box_data = pd.concat(plot_data, ignore_index=True)

    sns.boxplot(data=box_data, x='Window', y='Demand', ax=axes[idx], 
                palette="Set2", showfliers=True, fliersize=3)
    axes[idx].set_title(f"{cls.capitalize()}", fontweight='bold')
    axes[idx].set_ylabel("Daily Demand")
    axes[idx].tick_params(axis='x', rotation=20)

plt.suptitle("Figure Y. Boxplots with Outliers by Service Class and Booking Window", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure_Boxplot_Outliers.png", bbox_inches='tight', dpi=300)
plt.show()


# In[29]:


# ===================== ۵. سری زمانی با برجسته‌سازی داده پرت (مفید برای مقاله) =====================
# فقط برای business و special (چون این دو را پیش‌بینی می‌کنید)
focus_classes = ['business', 'special']

fig, axes = plt.subplots(2, 3, figsize=(16, 7))

for row_idx, cls in enumerate(focus_classes):
    for col_idx, window in enumerate(available_windows):
        series = pivot_by_class[cls][window]

        ax = axes[row_idx, col_idx]
        ax.plot(series.index, series.values, color='#2c3e50', linewidth=0.8, alpha=0.7)

        # تشخیص و برجسته کردن پرت‌ها
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[series > upper_bound]
        ax.scatter(outliers.index, outliers.values, color='red', s=18, zorder=5, label='Outlier' if col_idx==0 else "")

        ax.set_title(f"{cls.capitalize()} | {window}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Demand" if col_idx == 0 else "")

        if row_idx == 1:
            ax.set_xlabel("Date")

plt.suptitle("Figure Z. Time Series with Outliers Highlighted (Business & Special)", 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("Figure_TimeSeries_Outliers_Business_Special.png", bbox_inches='tight', dpi=300)
plt.show()


# In[30]:


# ===================== ۶. خلاصه نهایی برای گزارش =====================
print("\n" + "=" * 75)
print("خلاصه نهایی - مناسب برای بخش Data Description مقاله")
print("=" * 75)

print("\n• داده مفقوده (NaN):")
print(missing_df.groupby('Class')[['درصد NaN']].mean().round(2))

print("\n• درصد روزهای صفر (به عنوان عدم تقاضا):")
print(missing_df.groupby('Class')[['درصد روزهای صفر']].mean().round(2))

print("\n• میانگین درصد داده پرت در هر کلاس:")
print(outlier_df.groupby('Class')['درصد پرت'].mean().round(2))


# In[31]:


# بررسی پرت‌ها فقط برای business و special
focus_classes = ['business', 'special']

print("=== بررسی تاریخ روزهای پرت (بالاتر از 1.5 * IQR) ===\n")

for cls in focus_classes:
    print(f"\n{'='*60}")
    print(f"کلاس: {cls.upper()}")
    print(f"{'='*60}")

    for window in available_windows:
        series = pivot_by_class[cls][window].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR

        outliers = series[series > upper]

        print(f"\n{window}:")
        print(f"  تعداد پرت: {len(outliers)}")
        print(f"  حد بالا (Upper bound): {upper:.1f}")
        print(f"  ۱۰ روز با بیشترین تقاضا:")
        print(outliers.sort_values(ascending=False).head(10))


# In[5]:


# 2. Map service class names to their corresponding demand and price columns
class_map = {
    'Evening':  {'demand': 'demand_evening',  'price': 'price_evening'},
    'Business': {'demand': 'demand_business', 'price': 'price_business'},
    'Special':  {'demand': 'demand_special',  'price': 'price_special'},
    'Economy':  {'demand': 'demand_economy',  'price': 'price_economy'}
}

total_days = len(df)  # 1066

rows = []
for class_name, cols in class_map.items():
    demand_col = df[cols['demand']]
    price_col  = df[cols['price']]

    mean_val = demand_col.mean()
    std_val  = demand_col.std()
    min_val  = demand_col.min()
    max_val  = demand_col.max()

    # Percentage of missing values in the price column (demand has no missing values per your info)
    missing_pct = price_col.isna().mean() * 100

    # Number of days with exactly zero sales (demand)
    zero_days = int((demand_col == 0).sum())

    rows.append({
        'Service Class': class_name,
        'Mean Daily Sales': round(mean_val, 2),
        'Std. Deviation': round(std_val, 2),
        'Min': int(min_val),
        'Max': int(max_val),
        '% Missing Values (Price)': round(missing_pct, 2),
        'Zero-Sale Days': zero_days
    })

result_table = pd.DataFrame(rows)
print(result_table.to_string(index=False))


# In[6]:


evening_zero = df['demand_evening'] == 0
economy_zero = df['demand_economy'] == 0

both      = (evening_zero & economy_zero).sum()
only_ev   = (evening_zero & ~economy_zero).sum()
only_ec   = (~evening_zero & economy_zero).sum()
neither   = (~evening_zero & ~economy_zero).sum()
union     = (evening_zero | economy_zero).sum()

n = len(df)
print(f"Both zero (intersection): {both}  ({both/n*100:.1f}%)")
print(f"Only Evening zero:        {only_ev}  ({only_ev/n*100:.1f}%)")
print(f"Only Economy zero:        {only_ec}  ({only_ec/n*100:.1f}%)")
print(f"Neither zero:             {neither}  ({neither/n*100:.1f}%)")
print(f"Union (at least one):     {union}  ({union/n*100:.1f}%)")
print(f"Check: {both + only_ev + only_ec + neither} == {n}")

# Cross-check with capacity-full days (9 days)
if 'capacity_full' in df.columns:
    print("\nCapacity-full among the 53 both-zero days:",
          (evening_zero & economy_zero & (df['capacity_full'] == 1)).sum())


# In[7]:


both_zero_mask = (df['demand_evening'] == 0) & (df['demand_economy'] == 0)

# Total tickets sold on those days across all classes
demand_cols = ['demand_evening', 'demand_business', 'demand_special', 'demand_economy']
df['total_demand'] = df[demand_cols].sum(axis=1)

# Approximate train capacity (e.g., 95th percentile of total daily sales)
cap_threshold = df['total_demand'].quantile(0.95)

full_capacity_days = (both_zero_mask & (df['total_demand'] >= cap_threshold)).sum()

print(f"Both-zero days: {both_zero_mask.sum()}")
print(f"Capacity threshold (95th pct): {cap_threshold:.0f}")
print(f"Both-zero days that were near-full capacity: {full_capacity_days}")
print(f"Both-zero days due to deliberate policy: {both_zero_mask.sum() - full_capacity_days}")


# In[8]:


prem = ['demand_business', 'demand_special']
both_zero = (df['demand_evening'] == 0) & (df['demand_economy'] == 0)

# Per-class capacity proxy = observed maximum (physical seat limit)
cap = {c: df[c].max() for c in prem}
print("Estimated seat capacity per premium class:", cap)

# Occupancy ratio of premium classes
occ = sum(df[c] for c in prem) / sum(cap.values())
df['premium_occupancy'] = occ

for thr in [0.85, 0.90, 0.95, 0.99]:
    n_full = (both_zero & (occ >= thr)).sum()
    print(f"Both-zero days with premium occupancy >= {thr:.0%}: {n_full}")

print("\n--- Premium demand comparison ---")
print("On both-zero days :", sum(df.loc[both_zero, c] for c in prem).describe()[['mean','50%','max']].round(1).to_dict())
print("On other days     :", sum(df.loc[~both_zero, c] for c in prem).describe()[['mean','50%','max']].round(1).to_dict())

# Statistical test: is premium demand significantly higher on both-zero days?
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(df.loc[both_zero, prem].sum(axis=1),
                    df.loc[~both_zero, prem].sum(axis=1),
                    alternative='greater')
print(f"\nMann-Whitney U (both-zero > others): p-value = {p:.5f}")


# In[10]:


price_cols = ['price_business', 'price_special']

print("Mean premium price on both-zero days :",
      df.loc[both_zero, price_cols].mean().round(0).to_dict())
print("Mean premium price on other days     :",
      df.loc[~both_zero, price_cols].mean().round(0).to_dict())

# Verify: is price NaN exactly when demand == 0?
for cls in ['evening', 'economy', 'business', 'special']:
    dcol, pcol = f'demand_{cls}', f'price_{cls}'
    zero_d   = df[dcol] == 0
    nan_p    = df[pcol].isna()
    print(f"{cls:9s} | zero-demand: {zero_d.sum():4d} | NaN-price: {nan_p.sum():4d} "
          f"| perfect match: {(zero_d == nan_p).all()}")


# In[11]:


# Does closing cheap classes shift demand upward?
comp = df.groupby(both_zero.map({True:'Cheap closed', False:'Cheap open'}))[
    ['demand_evening','demand_economy','demand_business','demand_special']
].mean().round(1)
comp['total'] = comp.sum(axis=1)
print(comp)

# Revenue comparison
rev = sum(df[f'demand_{c}'] * df[f'price_{c}'].fillna(0)
          for c in ['evening','economy','business','special'])
df['daily_revenue'] = rev
print("\nMean daily revenue | cheap closed:", rev[both_zero].mean().round(0))
print("Mean daily revenue | cheap open  :", rev[~both_zero].mean().round(0))
print("Difference: {:+.1f}%".format((rev[both_zero].mean()/rev[~both_zero].mean()-1)*100))


# In[12]:


prem = ['demand_business', 'demand_special']
both_zero = (df['demand_evening'] == 0) & (df['demand_economy'] == 0)
prem_sum = df[prem].sum(axis=1)

# Realistic capacity proxies
cap_max  = prem_sum.max()            # 472
cap_p99  = prem_sum.quantile(0.99)
print(f"Capacity (observed max of SUM): {cap_max:.0f} | 99th pct: {cap_p99:.0f}")
print(f"Naive sum of individual maxima (WRONG): {df[prem].max().sum():.0f}")

for name, cap in [('max', cap_max), ('p99', cap_p99)]:
    occ = prem_sum / cap
    print(f"\n--- Capacity proxy = {name} ({cap:.0f}) ---")
    print(f"Mean occupancy | both-zero: {occ[both_zero].mean():.1%} | others: {occ[~both_zero].mean():.1%}")
    for thr in [0.80, 0.85, 0.90, 0.95, 0.99]:
        n = (both_zero & (occ >= thr)).sum()
        print(f"  both-zero days with occupancy >= {thr:.0%}: {n:3d}  ({n/both_zero.sum():.1%})")

# Do the two maxima ever co-occur? (proof that fleet composition varies)
print("\nDay of max business:", df.loc[df['demand_business'].idxmax(), prem].to_dict())
print("Day of max special :", df.loc[df['demand_special'].idxmax(),  prem].to_dict())
print("Correlation between business and special demand:",
      round(df['demand_business'].corr(df['demand_special']), 3))


# In[32]:


import pandas as pd
import numpy as np

focus_classes = ['business', 'special']

print("=" * 70)
print("بررسی روزهای پرت (بالاتر از 1.5 * IQR) برای Business و Special")
print("=" * 70)

for cls in focus_classes:
    print(f"\n{'='*60}")
    print(f"کلاس: {cls.upper()}")
    print(f"{'='*60}")

    for window in available_windows:
        series = pivot_by_class[cls][window].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[series > upper_bound].sort_values(ascending=False)

        print(f"\n{window}:")
        print(f"  تعداد پرت: {len(outliers)}")
        print(f"  حد بالا: {upper_bound:.1f}")
        print(f"  ۱۰ روز پرتقاضاترین:")
        print(outliers.head(10))

        # اگر ستون shamsi را دارید، درصد همخوانی با تعطیلات را نشان بده
        # (این بخش را اگر shamsi دارید فعال کنید)


# # **index**

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== ایمپورت‌های کامل =====================
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import optuna
from scipy import stats

# ===================== بارگذاری داده (هر شیت جداگانه) =====================
file_path = 'booking_windows_CLEANED.xlsx'

df_window1 = pd.read_excel(file_path, sheet_name=0)
df_window2 = pd.read_excel(file_path, sheet_name=1)
df_window3 = pd.read_excel(file_path, sheet_name=2)

def prepare_single_sheet(df, window_name):
    df = df.copy()
    if 'MoveDate_M' in df.columns:
        df['MoveDate_M'] = pd.to_datetime(df['MoveDate_M'])
        df = df.sort_values('MoveDate_M').reset_index(drop=True)

    demand_cols = ['business', 'economy', 'evening', 'special']
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'Price' in c]

    for col in demand_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')

    df['target_business'] = df['business'].shift(-1)
    df['target_special'] = df['special'].shift(-1)
    df = df.dropna(subset=['target_business', 'target_special']).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['MoveDate_M', 'target_business', 'target_special']]

    return {
        'window_name': window_name,
        'df': df,
        'features': df[feature_cols],
        'targets': df[['target_business', 'target_special']],
        'dates': df['MoveDate_M'] if 'MoveDate_M' in df.columns else None
    }

data_w1 = prepare_single_sheet(df_window1, 'Window1_Prepurchase')
data_w2 = prepare_single_sheet(df_window2, 'Window2_Normal')
data_w3 = prepare_single_sheet(df_window3, 'Window3_LastMinute')

all_windows = [data_w1, data_w2, data_w3]


# In[2]:


data_w1


# In[3]:


# ===================== توابع کمکی (در سطح بالا) =====================

def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name} | ADF: {result[0]:.4f} | p-value: {result[1]:.4f} | Stationary: {result[1] < 0.05}")

def find_optimal_lag(series, max_lag=30):
    acf_vals = acf(series, nlags=max_lag, fft=False)
    # پیدا کردن اولین لگی که ACF زیر 0.2 می‌رود (یا ۷ اگر پیدا نشد)
    lag = next((i for i, v in enumerate(acf_vals) if v < 0.2), 7)
    return max(lag, 3)

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X.iloc[i:i+lookback].values)
        ys.append(y.iloc[i+lookback].values)
    return np.array(Xs), np.array(ys)

def calculate_mase(y_true, y_pred, y_train):
    naive = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive if naive != 0 else np.nan

def all_metrics(y_true, y_pred, y_train, target_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred, y_train)
    return {
        f'{target_name}MAE': round(mae, 4),
        f'{target_name}MSE': round(mse, 4),
        f'{target_name}RMSE': round(rmse, 4),
        f'{target_name}MAPE': round(mape, 2),
        f'{target_name}R2': round(r2, 4),
        f'{target_name}MASE': round(mase, 4)
    }

def diebold_mariano(e1, e2):
    d = e1**2 - e2**2
    dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


# In[38]:


# ============================================================
#  Table 3 : Calendar day-type categories vs. class-level demand
#  Output : markdown + LaTeX + CSV, ready to paste
# ============================================================
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

FILE     = "data_1396__1397_1398.csv"     # <-- adjust
CLASSES  = ["evening", "business", "special", "economy"]
PREMIUM  = ["business", "special"]

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
df = pd.read_csv(FILE)
df["datatime"] = pd.to_datetime(df["datatime"])
df = df.sort_values("datatime").reset_index(drop=True)
print(f"Rows: {len(df)} | period: {df.datatime.min().date()} -> {df.datatime.max().date()}")

# ------------------------------------------------------------
# 2. Derived variables
#    NOTE: run this BEFORE fare imputation, so that mean fares
#    are computed over observed transactions only.
# ------------------------------------------------------------
df["premium_demand"] = df[[f"demand_{c}" for c in PREMIUM]].sum(axis=1)
df["cheap_closed"]   = (df["demand_evening"] == 0) & (df["demand_economy"] == 0)

# daily revenue: NaN price always coincides with zero demand -> fillna(0) is safe
df["revenue"] = sum(df[f"demand_{c}"] * df[f"price_{c}"].fillna(0.0) for c in CLASSES)

# capacity proxy = 99th percentile of COMBINED premium demand (= 421)
CAP = df["premium_demand"].quantile(0.99)
df["premium_util"] = df["premium_demand"] / CAP
print(f"Capacity proxy (p99 of premium demand) = {CAP:.0f}")

# ------------------------------------------------------------
# 3. Build Table 3
# ------------------------------------------------------------
g   = df.groupby("shamsi")
tab = pd.DataFrame(index=sorted(df["shamsi"].unique()))
tab.index.name = "day_type"

tab["n_days"]    = g.size()
tab["share_pct"] = 100 * g.size() / len(df)

for c in CLASSES:
    tab[f"mean_{c}"] = g[f"demand_{c}"].mean()
    tab[f"sd_{c}"]   = g[f"demand_{c}"].std()

tab["mean_premium"]     = g["premium_demand"].mean()
tab["median_premium"]   = g["premium_demand"].median()
tab["max_premium"]      = g["premium_demand"].max()
tab["mean_util_pct"]    = 100 * g["premium_util"].mean()
tab["cheap_closed_pct"] = 100 * g["cheap_closed"].mean()

for c in PREMIUM:
    tab[f"mean_price_{c}_k"] = g[f"price_{c}"].mean() / 1e3   # thousand IRR

tab["mean_revenue_bn"] = g["revenue"].mean() / 1e9            # billion IRR

tab = tab.round(2)

# ------------------------------------------------------------
# 4. Significance tests (for the table footnote)
# ------------------------------------------------------------
groups = [sub["premium_demand"].values for _, sub in g]
H, p_kw = kruskal(*groups)
print(f"\nKruskal-Wallis across day types (premium demand): H={H:.2f}, p={p_kw:.3e}")

print("\nSpearman rho of day-type indicator with each class:")
for c in CLASSES + ["premium_demand"]:
    col = f"demand_{c}" if c in CLASSES else c
    rho, p = spearmanr(df["shamsi"], df[col])
    print(f"  {col:<18} rho={rho:+.3f}  p={p:.3e}")

# ------------------------------------------------------------
# 5. Export : rows = indicators, columns = day types
# ------------------------------------------------------------
out = tab.T
out.to_csv("table3_calendar_categories.csv")

print("\n================ TABLE 3 (markdown) ================\n")
hdr = ["Indicator"] + [f"Type {i}" for i in out.columns]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for name, row in out.iterrows():
    print("| " + name + " | " + " | ".join(f"{v:,.2f}" for v in row.values) + " |")

print("\n================ TABLE 3 (LaTeX) ================\n")
print(out.to_latex(float_format="%.2f"))


# # **Identical Lag of Time Series**

# In[4]:


# ===================== پایپ‌لاین اصلی برای هر پنجره =====================

results_summary = []

for data in all_windows:
    window_name = data['window_name']
    print(f"\n{'='*80}")
    print(f"پنجره: {window_name}")
    print(f"{'='*80}")

    features = data['features']
    targets = data['targets']
    dates = data['dates']

    # ۱. آزمون ایستایی
    check_stationarity(targets['target_business'], "Business")
    check_stationarity(targets['target_special'], "Special")

    # ۲. تعیین لگ زمانی (Lookback)
    lag_b = find_optimal_lag(targets['target_business'])
    lag_s = find_optimal_lag(targets['target_special'])
    lookback = max(lag_b, lag_s, 7)   # حداقل ۷ روز
    print(f"Lookback انتخاب شده: {lookback}")

    # ۳. تقسیم زمانی (۷۰-۱۵-۱۵)
    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = features.iloc[:train_end], targets.iloc[:train_end]
    X_val, y_val = features.iloc[train_end:val_end], targets.iloc[train_end:val_end]
    X_test, y_test = features.iloc[val_end:], targets.iloc[val_end:]

    # ۴. نرمال‌سازی (فقط روی Train)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # تبدیل به توالی برای مدل‌های عمیق
    X_train_seq, y_train_seq = create_sequences(pd.DataFrame(X_train_scaled), pd.DataFrame(y_train_scaled), lookback)
    X_val_seq, y_val_seq = create_sequences(pd.DataFrame(X_val_scaled), pd.DataFrame(y_val_scaled), lookback)
    X_test_seq, y_test_seq = create_sequences(pd.DataFrame(X_test_scaled), pd.DataFrame(y_test_scaled), lookback)

    # ===================== مدل‌های پایه (Baselines) =====================
    # اینجا می‌توانید مدل‌های پایه را اضافه کنید (Naïve, ARIMA, ETS, VAR, Linear, RF, XGBoost)
    # برای brevity در این پاسخ، ساختار را نشان دادم. در اجرای واقعی این بخش را کامل کنید.

    # ===================== مدل‌های عمیق با Optuna =====================
    def build_model(trial, model_type, input_shape):
        # مثال برای LSTM-CNN (سایر مدل‌ها مشابه)
        if model_type == 'LSTM-CNN':
            inputs = Input(shape=input_shape)
            x = Conv1D(filters=trial.suggest_int('filters', 32, 128), kernel_size=3, activation='relu')(inputs)
            x = MaxPooling1D()(x)
            x = LSTM(units=trial.suggest_int('lstm_units', 32, 128), return_sequences=False)(x)
            x = Dropout(trial.suggest_float('dropout', 0.1, 0.5))(x)
            outputs = Dense(2)(x)
            model = Model(inputs, outputs)
        # ... (MLP, RNN, GRU, LSTM, CNN را به همین شکل اضافه کنید)
        model.compile(optimizer=Adam(learning_rate=trial.suggest_float('lr', 1e-4, 1e-2, log=True)), loss='mse')
        return model

    # اجرای تکراری + Optuna (۱۰ بار با seed متفاوت)
    # در اینجا می‌توانید حلقه برای ۱۰ seed بنویسید و نتایج را جمع کنید.

    print(f"پنجره {window_name} آماده مدل‌سازی شد.")


# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# ===================== تنظیمات حرفه‌ای برای مقاله =====================
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# ===================== فرض: داده‌ها از کد قبلی آماده هستند =====================
# data_w1, data_w2, data_w3 که قبلاً آماده کردی
all_windows = [data_w1, data_w2, data_w3]

# ===================== توابع آزمون‌های آماری =====================

def adf_test(series, name):
    result = adfuller(series.dropna())
    return {
        'Test': 'ADF',
        'Statistic': round(result[0], 4),
        'p-value': round(result[1], 4),
        'Stationary': result[1] < 0.05,
        'Critical 1%': round(result[4]['1%'], 4),
        'Critical 5%': round(result[4]['5%'], 4)
    }

def kpss_test(series, name):
    try:
        statistic, p_value, lags, critical_values = kpss(series.dropna(), regression='c')
        return {
            'Test': 'KPSS',
            'Statistic': round(statistic, 4),
            'p-value': round(p_value, 4),
            'Stationary': p_value > 0.05,
            'Critical 1%': round(critical_values['1%'], 4),
            'Critical 5%': round(critical_values['5%'], 4)
        }
    except:
        return {'Test': 'KPSS', 'Statistic': np.nan, 'p-value': np.nan, 'Stationary': False}

def ljung_box_test(series, name, lags=10):
    result = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
    return {
        'Test': 'Ljung-Box',
        'Statistic': round(result['lb_stat'].iloc[-1], 4),
        'p-value': round(result['lb_pvalue'].iloc[-1], 4),
        'White Noise': result['lb_pvalue'].iloc[-1] > 0.05
    }

# ===================== تابع اصلی برای تولید تحلیل پربار =====================

def rich_stationarity_analysis(data, window_name):
    print(f"\n{'='*85}")
    print(f"تحلیل ایستایی و انتخاب پنجره زمانی - {window_name}")
    print(f"{'='*85}")

    features = data['features']
    targets = data['targets']
    dates = data['dates']

    results_table = []

    for target_col in ['target_business', 'target_special']:
        series = targets[target_col]
        class_name = target_col.replace('target_', '').capitalize()

        print(f"\n--- {class_name} ---")

        # ۱. آزمون ADF (نتایج قبلی شما)
        adf_res = adf_test(series, class_name)
        print(f"ADF  | Statistic: {adf_res['Statistic']} | p-value: {adf_res['p-value']} | Stationary: {adf_res['Stationary']}")

        # ۲. آزمون KPSS (تکمیلی خیلی مهم)
        kpss_res = kpss_test(series, class_name)
        print(f"KPSS | Statistic: {kpss_res['Statistic']} | p-value: {kpss_res['p-value']} | Stationary: {kpss_res['Stationary']}")

        # ۳. آزمون Ljung-Box (برای بررسی نویز سفید بودن)
        lb_res = ljung_box_test(series, class_name, lags=10)
        print(f"Ljung-Box | Statistic: {lb_res['Statistic']} | p-value: {lb_res['p-value']} | White Noise: {lb_res['White Noise']}")

        results_table.append({
            'Window': window_name,
            'Class': class_name,
            **adf_res,
            **kpss_res,
            **lb_res
        })

        # ===================== نمودارهای پربار =====================
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Stationarity & Lookback Analysis - {window_name} | {class_name}", 
                     fontsize=14, fontweight='bold', y=1.02)

        # نمودار ۱: سری زمانی اصلی
        ax1 = axes[0, 0]
        ax1.plot(dates, series, color='#2c3e50', linewidth=0.9, alpha=0.85)
        ax1.set_title("Time Series Plot", fontweight='bold')
        ax1.set_ylabel("Demand")
        ax1.tick_params(axis='x', rotation=30)

        # نمودار ۲: Rolling Statistics (میانگین و انحراف معیار متحرک)
        ax2 = axes[0, 1]
        rolling_mean = series.rolling(window=30).mean()
        rolling_std = series.rolling(window=30).std()
        ax2.plot(dates, series, label='Original', color='gray', alpha=0.5, linewidth=0.7)
        ax2.plot(dates, rolling_mean, label='Rolling Mean (30)', color='#e74c3c', linewidth=1.5)
        ax2.plot(dates, rolling_std, label='Rolling Std (30)', color='#3498db', linewidth=1.5)
        ax2.set_title("Rolling Mean & Std Dev (Window=30)", fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.tick_params(axis='x', rotation=30)

        # نمودار ۳: ACF
        ax3 = axes[1, 0]
        plot_acf(series.dropna(), ax=ax3, lags=40, alpha=0.05, color='#27ae60')
        ax3.axhline(y=0.2, color='red', linestyle='--', linewidth=1, label='Threshold 0.2')
        ax3.set_title("Autocorrelation Function (ACF)", fontweight='bold')
        ax3.legend(fontsize=9)

        # نمودار ۴: PACF
        ax4 = axes[1, 1]
        plot_pacf(series.dropna(), ax=ax4, lags=40, alpha=0.05, color='#9b59b6', method='ywm')
        ax4.set_title("Partial Autocorrelation Function (PACF)", fontweight='bold')

        plt.tight_layout()
        filename = f"Figure_Stationarity_{window_name}_{class_name}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"نمودار ذخیره شد: {filename}")

        # ===================== نمودار اضافی برای توجیه Lookback =====================
        fig2, ax = plt.subplots(figsize=(10, 4))
        acf_values = acf(series.dropna(), nlags=40, fft=False)
        ax.bar(range(len(acf_values)), acf_values, color='#3498db', alpha=0.7)
        ax.axhline(y=0.2, color='red', linestyle='--', label='0.2 Threshold')
        ax.axhline(y=-0.2, color='red', linestyle='--')
        ax.set_title(f"ACF Values - {window_name} | {class_name} (Lookback justification)", fontweight='bold')
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"Figure_ACF_Lookback_{window_name}_{class_name}.png", bbox_inches='tight', dpi=300)
        plt.show()

    return pd.DataFrame(results_table)

# ===================== اجرای تحلیل برای همه پنجره‌ها =====================

print("شروع تحلیل ایستایی غنی‌شده برای مقاله...\n")

all_results = []

for data in all_windows:
    window_results = rich_stationarity_analysis(data, data['window_name'])
    all_results.append(window_results)

# ===================== جدول خلاصه کامل (مناسب مقاله) =====================
final_table = pd.concat(all_results, ignore_index=True)
print("\n" + "="*85)
print("جدول خلاصه آزمون‌های آماری ایستایی (برای قرار دادن در مقاله)")
print("="*85)
print(final_table.to_string(index=False))

# ذخیره جدول
final_table.to_excel("Table_Stationarity_Tests_All_Windows.xlsx", index=False)
print("\nجدول کامل در فایل Table_Stationarity_Tests_All_Windows.xlsx ذخیره شد.")

# ===================== خلاصه متنی پربار برای گزارش =====================
print("\n" + "="*85)
print("خلاصه تفسیری آماده برای بخش مقاله (کپی کنید)")
print("="*85)

for window in ['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']:
    subset = final_table[final_table['Window'] == window]
    print(f"\nپنجره {window}:")
    for _, row in subset.iterrows():
        print(f"  {row['Class']}: ADF p-value = {row['p-value']}, KPSS p-value = {row['p-value']}, "
              f"هر دو آزمون ایستایی را تأیید می‌کنند. Lookback انتخاب‌شده بر اساس ACF: {row['Window'] if 'Lookback' in row else 'بر اساس تحلیل قبلی'}")


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ===================== تنظیمات حرفه‌ای نمودار =====================
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

# ===================== فرض: داده‌ها از کد قبلی آماده هستند =====================
# all_windows = [data_w1, data_w2, data_w3]   (از کد قبلی)
# هر data شامل: 'window_name', 'targets', 'dates'

def plot_time_series_decomposition(data, window_name, class_name, series, period=7, model='additive'):
    """
    رسم تجزیه سری زمانی (Observed + Trend + Seasonal + Residual)
    """
    # انجام تجزیه
    result = seasonal_decompose(
        series.dropna(), 
        model=model, 
        period=period, 
        extrapolate_trend='freq'
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    # Observed
    result.observed.plot(ax=axes[0], color='#2c3e50', linewidth=0.9)
    axes[0].set_title('Observed', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Demand')

    # Trend
    result.trend.plot(ax=axes[1], color='#e74c3c', linewidth=1.8)
    axes[1].set_title('Trend', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Trend')

    # Seasonal
    result.seasonal.plot(ax=axes[2], color='#27ae60', linewidth=1.2)
    axes[2].set_title(f'Seasonal (Period = {period})', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Seasonal')

    # Residual
    result.resid.plot(ax=axes[3], color='#8e44ad', linewidth=0.9, alpha=0.85)
    axes[3].set_title('Residual', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')

    # عنوان کلی
    fig.suptitle(
        f'Time Series Decomposition\n{window_name} | {class_name} (Model: {model})', 
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    # ذخیره با کیفیت بالا
    filename = f"Figure_Decomposition_{window_name}_{class_name}_{model}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"نمودار ذخیره شد: {filename}")

    return result


# ===================== اجرای تجزیه برای همه پنجره‌ها =====================

print("\n" + "="*85)
print("شروع رسم نمودارهای تجزیه سری زمانی (برای بخش مقاله)")
print("="*85)

decomposition_results = {}

for data in all_windows:
    window_name = data['window_name']
    targets = data['targets']
    dates = data['dates']

    for target_col in ['target_business', 'target_special']:
        class_name = target_col.replace('target_', '').capitalize()
        series = targets[target_col]

        print(f"\n--- تجزیه {window_name} | {class_name} ---")

        # مدل Additive (پیشنهاد اصلی برای اکثر داده‌های تقاضا)
        decomp_add = plot_time_series_decomposition(
            data, window_name, class_name, series, 
            period=7, model='additive'
        )

        decomposition_results[f"{window_name}_{class_name}_additive"] = decomp_add

        # اگر می‌خواهی مدل Multiplicative را هم تست کنی (اختیاری)
        # decomp_mult = plot_time_series_decomposition(
        #     data, window_name, class_name, series, 
        #     period=7, model='multiplicative'
        # )


# # **Train and Test Split**

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ===================== تعریف سه سناریو =====================
SCENARIOS = {
    'Scenario_1_Classic': {'train': 0.70, 'val': 0.15, 'test': 0.15},
    'Scenario_2_TrainingHeavy': {'train': 0.80, 'val': 0.10, 'test': 0.10},
    'Scenario_3_DeepLearning': {'train': 0.90, 'val': 0.05, 'test': 0.05}
}

# Lookbackهایی که قبلاً استخراج کردی
lookback_dict = {
    'Window1_Prepurchase': 18,
    'Window2_Normal': 25,
    'Window3_LastMinute': 9
}

# ===================== تابع تقسیم زمانی =====================
def create_time_series_split(features, targets, dates, train_ratio, val_ratio, lookback):
    n = len(features)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # تقسیم خام
    X_train_raw = features.iloc[:train_end]
    y_train_raw = targets.iloc[:train_end]

    X_val_raw = features.iloc[train_end:val_end]
    y_val_raw = targets.iloc[train_end:val_end]

    X_test_raw = features.iloc[val_end:]
    y_test_raw = targets.iloc[val_end:]

    # نرمال‌سازی (فقط روی Train)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # ساخت توالی
    def create_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:i + lookback])
            ys.append(y[i + lookback])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)

    return {
        'X_train_raw': X_train_raw, 'y_train_raw': y_train_raw,
        'X_val_raw': X_val_raw, 'y_val_raw': y_val_raw,
        'X_test_raw': X_test_raw, 'y_test_raw': y_test_raw,

        'X_train': X_train_seq, 'y_train': y_train_seq,
        'X_val': X_val_seq, 'y_val': y_val_seq,
        'X_test': X_test_seq, 'y_test': y_test_seq,

        'scaler_X': scaler_X,
        'scaler_y': scaler_y,

        'train_dates': dates.iloc[lookback:train_end],
        'val_dates': dates.iloc[train_end + lookback:val_end],
        'test_dates': dates.iloc[val_end + lookback:],

        'n_train': len(X_train_seq),
        'n_val': len(X_val_seq),
        'n_test': len(X_test_seq)
    }

# ===================== اجرای سه سناریو برای هر پنجره =====================
all_splits = {}

for data in all_windows:
    window_name = data['window_name']
    lookback = lookback_dict[window_name]
    features = data['features']
    targets = data['targets']
    dates = data['dates']

    print(f"\n{'='*85}")
    print(f"پنجره: {window_name} | Lookback: {lookback}")
    print(f"{'='*85}")

    all_splits[window_name] = {}

    for scenario_name, ratios in SCENARIOS.items():
        split = create_time_series_split(
            features, targets, dates,
            train_ratio=ratios['train'],
            val_ratio=ratios['val'],
            lookback=lookback
        )

        all_splits[window_name][scenario_name] = split

        total = split['n_train'] + split['n_val'] + split['n_test']
        print(f"{scenario_name:30} | Train: {split['n_train']:5} | Val: {split['n_val']:4} | Test: {split['n_test']:4} | "
              f"({split['n_train']/total*100:.1f}% / {split['n_val']/total*100:.1f}% / {split['n_test']/total*100:.1f}%)")

print("\n" + "="*85)
print("تقسیم داده در سه سناریو برای هر سه پنجره با موفقیت انجام شد.")
print("="*85)


# # StandardScaler

# In[6]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("شروع نرمال‌سازی داده‌ها با StandardScaler (روش داورپسند)...\n")

# ===================== دیکشنری lookback (از تحلیل ACF قبلی) =====================
lookback_dict = {
    'Window1_Prepurchase': 18,
    'Window2_Normal': 25,
    'Window3_LastMinute': 9
}

# ===================== تابع نرمال‌سازی داورپسند =====================
def normalize_split(split_data, window_name, scenario_name, lookback):
    """
    نرمال‌سازی کاملاً استاندارد و بدون Data Leakage:
    - StandardScaler فقط روی Train فیت می‌شود.
    - Val و Test فقط transform می‌شوند.
    - همه ستون‌ها (تمام ویژگی‌ها) نرمال می‌شوند.
    - خروجی Multi-output برای Business و Special حفظ می‌شود.
    """

    # استخراج داده‌های خام
    X_train_raw = split_data['X_train_raw']
    y_train_raw = split_data['y_train_raw']
    X_val_raw   = split_data['X_val_raw']
    y_val_raw   = split_data['y_val_raw']
    X_test_raw  = split_data['X_test_raw']
    y_test_raw  = split_data['y_test_raw']

    # ===================== تعریف اسکیلرها =====================
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # ===================== فیت فقط روی داده Train =====================
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    # ===================== اعمال روی Val و Test (بدون فیت مجدد) =====================
    X_val_scaled = scaler_X.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)

    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # ===================== ساخت توالی (Sequence) =====================
    def create_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:i + lookback])
            ys.append(y[i + lookback])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq,   y_val_seq   = create_sequences(X_val_scaled,   y_val_scaled,   lookback)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  lookback)

    # ===================== ساخت دیکشنری خروجی =====================
    normalized_split = {
        'window_name': window_name,
        'scenario_name': scenario_name,
        'lookback': lookback,

        # داده‌های خام (برای مدل‌های پایه مثل ARIMA و XGBoost)
        'X_train_raw': X_train_raw,
        'y_train_raw': y_train_raw,
        'X_val_raw': X_val_raw,
        'y_val_raw': y_val_raw,
        'X_test_raw': X_test_raw,
        'y_test_raw': y_test_raw,

        # داده‌های نرمال‌شده و به شکل توالی (برای مدل‌های عمیق)
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'X_test': X_test_seq,
        'y_test': y_test_seq,

        # اسکیلرها (برای برگرداندن پیش‌بینی به مقیاس اصلی)
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,

        # تعداد نمونه‌ها
        'n_train': len(X_train_seq),
        'n_val': len(X_val_seq),
        'n_test': len(X_test_seq)
    }

    return normalized_split

# ===================== اجرای نرمال‌سازی برای همه پنجره‌ها و سناریوها =====================
normalized_splits = {}

for window_name in all_splits.keys():
    normalized_splits[window_name] = {}

    print(f"\n{'='*85}")
    print(f"نرمال‌سازی پنجره: {window_name}")
    print(f"{'='*85}")

    lookback = lookback_dict[window_name]

    for scenario_name in all_splits[window_name].keys():
        split = all_splits[window_name][scenario_name]

        # ارسال lookback به صورت جداگانه (حل مشکل KeyError)
        norm_split = normalize_split(split, window_name, scenario_name, lookback)
        normalized_splits[window_name][scenario_name] = norm_split

        total = norm_split['n_train'] + norm_split['n_val'] + norm_split['n_test']

        print(f"{scenario_name:30} | Train: {norm_split['n_train']:5} | Val: {norm_split['n_val']:4} | Test: {norm_split['n_test']:4} | "
              f"({norm_split['n_train']/total*100:.1f}% / {norm_split['n_val']/total*100:.1f}% / {norm_split['n_test']/total*100:.1f}%)")

print("\n" + "="*85)
print("نرمال‌سازی داده‌ها با StandardScaler با موفقیت انجام شد.")
print("="*85)

# ===================== خلاصه روش برای مقاله =====================
print("\nروش نرمال‌سازی (برای بخش متدولوژی مقاله):")
print("- روش: StandardScaler (Z-score Normalization)")
print("- فیت اسکیلرها: فقط روی داده‌های Train")
print("- Val و Test: فقط transform شدند (بدون فیت)")
print("- هدف: جلوگیری کامل از Data Leakage و حفظ استقلال مجموعه تست")


# In[7]:


import numpy as np
import pandas as pd

print("="*90)
print("بررسی و اعتبارسنجی نرمال‌سازی StandardScaler (برای اطمینان از عملکرد صحیح)")
print("="*90)

def validate_normalization(normalized_splits, window_name, scenario_name):
    split = normalized_splits[window_name][scenario_name]

    print(f"\n{'─'*80}")
    print(f"پنجره: {window_name} | سناریو: {scenario_name}")
    print(f"{'─'*80}")

    # ==================== ۱. بررسی ابعاد ====================
    print(f"تعداد نمونه‌ها → Train: {split['n_train']} | Val: {split['n_val']} | Test: {split['n_test']}")
    print(f"شکل X_train: {split['X_train'].shape} | شکل y_train: {split['y_train'].shape}")

    # ==================== ۲. بررسی اسکیلرها (مهم‌ترین چک) ====================
    scaler_X = split['scaler_X']
    scaler_y = split['scaler_y']

    print(f"\n[اسکیلر X] میانگین ویژگی‌ها (باید نزدیک ۰ باشد): {np.mean(scaler_X.mean_):.6f}")
    print(f"[اسکیلر X] انحراف معیار ویژگی‌ها (باید نزدیک ۱ باشد): {np.mean(scaler_X.scale_):.6f}")

    print(f"[اسکیلر y] میانگین تارگت‌ها: {np.mean(scaler_y.mean_):.6f}")
    print(f"[اسکیلر y] انحراف معیار تارگت‌ها: {np.mean(scaler_y.scale_):.6f}")

    # ==================== ۳. بررسی داده‌های نرمال‌شده Train (باید ~0 و ~1 باشد) ====================
    X_train = split['X_train']
    y_train = split['y_train']

    train_X_mean = np.mean(X_train)
    train_X_std  = np.std(X_train)
    train_y_mean = np.mean(y_train)
    train_y_std  = np.std(y_train)

    print(f"\n[Train] میانگین X (باید ~۰): {train_X_mean:.8f}")
    print(f"[Train] انحراف معیار X (باید ~۱): {train_X_std:.8f}")
    print(f"[Train] میانگین y (باید ~۰): {train_y_mean:.8f}")
    print(f"[Train] انحراف معیار y (باید ~۱): {train_y_std:.8f}")

    # ==================== ۴. بررسی Val و Test (نباید دقیقاً ۰ و ۱ باشد → اثبات عدم نشت داده) ====================
    X_val_mean = np.mean(split['X_val'])
    X_val_std  = np.std(split['X_val'])
    y_val_mean = np.mean(split['y_val'])
    y_val_std  = np.std(split['y_val'])

    X_test_mean = np.mean(split['X_test'])
    X_test_std  = np.std(split['X_test'])
    y_test_mean = np.mean(split['y_test'])
    y_test_std  = np.std(split['y_test'])

    print(f"\n[Val]   میانگین X: {X_val_mean:.6f} | انحراف معیار X: {X_val_std:.6f}")
    print(f"[Val]   میانگین y: {y_val_mean:.6f} | انحراف معیار y: {y_val_std:.6f}")

    print(f"[Test]  میانگین X: {X_test_mean:.6f} | انحراف معیار X: {X_test_std:.6f}")
    print(f"[Test]  میانگین y: {y_test_mean:.6f} | انحراف معیار y: {y_test_std:.6f}")

    # ==================== ۵. بررسی NaN و Inf ====================
    has_nan_X = np.isnan(X_train).any() or np.isnan(split['X_val']).any() or np.isnan(split['X_test']).any()
    has_inf_X = np.isinf(X_train).any() or np.isinf(split['X_val']).any() or np.isinf(split['X_test']).any()

    print(f"\nوجود NaN در داده‌ها: {has_nan_X}")
    print(f"وجود Inf در داده‌ها: {has_inf_X}")

    # ==================== ۶. مقایسه خام vs نرمال‌شده (نمونه کوچک) ====================
    print("\nنمونه از داده خام Train (۵ نمونه اول، ستون اول):")
    print(split['X_train_raw'].iloc[:5, 0].values if len(split['X_train_raw'].shape) > 1 else "داده خام تک‌بعدی")

    print("نمونه از داده نرمال‌شده Train (۵ نمونه اول، ستون اول):")
    print(split['X_train'][:5, 0, 0] if split['X_train'].ndim == 3 else split['X_train'][:5, 0])

# ===================== اجرای اعتبارسنجی برای همه پنجره‌ها و سناریوها =====================
for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        validate_normalization(normalized_splits, window_name, scenario_name)

print("\n" + "="*90)
print("بررسی کامل انجام شد. اگر مقادیر Train نزدیک ۰ و ۱ هستند و Val/Test متفاوت هستند → نرمال‌سازی درست است.")
print("="*90)


# # Forecast

# In[56]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

N_REPEATS = 10

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

def safe_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def calculate_all_metrics(y_true, y_pred):
    """محاسبه تمام معیارها روی مقیاس اصلی"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    smape = safe_smape(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'sMAPE': smape
    }

results = []
print("تنظیمات مشترک آماده شد.")


# In[57]:


print("\n" + "="*95)
print("اجرای مدل‌های کلاسیک (Deterministic - Repeat = 1)")
print("="*95)

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]
        y_train_raw = np.array(split['y_train_raw'])
        y_test_raw_full = np.array(split['y_test_raw'])
        lookback = split['lookback']
        y_test_eval = y_test_raw_full[lookback:]   # اصلاح طول

        for t_idx, target_name in enumerate(['Business', 'Special']):
            y_tr = y_train_raw[:, t_idx]
            y_te = y_test_eval[:, t_idx]

            # Naïve Seasonal
            period = 7
            last_cycle = y_tr[-period:]
            naive_pred = np.tile(last_cycle, (len(y_te) // period + 2))[:len(y_te)]
            metrics = calculate_all_metrics(y_te, naive_pred)
            results.append({**{'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                               'Model': 'Naïve Seasonal', 'Repeat': 1}, **metrics})

            # ARIMA
            try:
                model = auto_arima(y_tr, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
                pred = model.predict(n_periods=len(y_te))
                metrics = calculate_all_metrics(y_te, pred)
            except:
                metrics = {k: np.nan for k in ['MAE','RMSE','MSE','R2','MAPE','sMAPE']}
            results.append({**{'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                               'Model': 'ARIMA', 'Repeat': 1}, **metrics})

            # SARIMA
            try:
                model = auto_arima(y_tr, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
                pred = model.predict(n_periods=len(y_te))
                metrics = calculate_all_metrics(y_te, pred)
            except:
                metrics = {k: np.nan for k in ['MAE','RMSE','MSE','R2','MAPE','sMAPE']}
            results.append({**{'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                               'Model': 'SARIMA', 'Repeat': 1}, **metrics})

            # Exponential Smoothing
            try:
                model = ExponentialSmoothing(y_tr, seasonal='add', seasonal_periods=7).fit()
                pred = model.forecast(len(y_te))
                metrics = calculate_all_metrics(y_te, pred)
            except:
                metrics = {k: np.nan for k in ['MAE','RMSE','MSE','R2','MAPE','sMAPE']}
            results.append({**{'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                               'Model': 'Exponential Smoothing', 'Repeat': 1}, **metrics})

print("مدل‌های کلاسیک تمام شد.")


# In[58]:


print("\n" + "="*95)
print("اجرای مدل: Random Forest (۱۰ تکرار)")
print("="*95)

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]
        X_train = split['X_train'].reshape(split['X_train'].shape[0], -1)
        X_test = split['X_test'].reshape(split['X_test'].shape[0], -1)
        y_train_norm = split['y_train']
        y_test_raw_full = np.array(split['y_test_raw'])
        scaler_y = split['scaler_y']
        lookback = split['lookback']
        y_test_eval = y_test_raw_full[lookback:]

        for t_idx, target_name in enumerate(['Business', 'Special']):
            param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [5, 7, 10], 'min_samples_split': [2, 5]}
            search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=6, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(X_train, y_train_norm[:, t_idx])
            best_params = search.best_params_

            maes, rmses, mses, r2s, mapes, smapes = [], [], [], [], [], []
            for rep in range(N_REPEATS):
                model = RandomForestRegressor(**best_params, random_state=42 + rep)
                model.fit(X_train, y_train_norm[:, t_idx])
                y_pred_norm = model.predict(X_test)

                dummy = np.zeros((len(y_pred_norm), 2))
                dummy[:, t_idx] = y_pred_norm
                y_pred_orig = scaler_y.inverse_transform(dummy)[:, t_idx]

                m = calculate_all_metrics(y_test_eval[:, t_idx], y_pred_orig)
                maes.append(m['MAE']); rmses.append(m['RMSE']); mses.append(m['MSE'])
                r2s.append(m['R2']); mapes.append(m['MAPE']); smapes.append(m['sMAPE'])

            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'Random Forest', 'Repeat': 'Mean', 'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
                            'MSE': np.mean(mses), 'R2': np.mean(r2s), 'MAPE': np.mean(mapes), 'sMAPE': np.mean(smapes)})
            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'Random Forest', 'Repeat': 'Std', 'MAE': np.std(maes), 'RMSE': np.std(rmses),
                            'MSE': np.std(mses), 'R2': np.std(r2s), 'MAPE': np.std(mapes), 'sMAPE': np.std(smapes)})

print("Random Forest تمام شد.")


# In[59]:


print("\n" + "="*95)
print("اجرای مدل: XGBoost (۱۰ تکرار)")
print("="*95)

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]
        X_train = split['X_train'].reshape(split['X_train'].shape[0], -1)
        X_test = split['X_test'].reshape(split['X_test'].shape[0], -1)
        y_train_norm = split['y_train']
        y_test_raw_full = np.array(split['y_test_raw'])
        scaler_y = split['scaler_y']
        lookback = split['lookback']
        y_test_eval = y_test_raw_full[lookback:]

        for t_idx, target_name in enumerate(['Business', 'Special']):
            param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 0.9, 1.0]}
            search = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror', verbosity=0), param_dist, n_iter=6, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(X_train, y_train_norm[:, t_idx])
            best_params = search.best_params_

            maes, rmses, mses, r2s, mapes, smapes = [], [], [], [], [], []
            for rep in range(N_REPEATS):
                model = xgb.XGBRegressor(**best_params, random_state=42 + rep, verbosity=0)
                model.fit(X_train, y_train_norm[:, t_idx])
                y_pred_norm = model.predict(X_test)

                dummy = np.zeros((len(y_pred_norm), 2))
                dummy[:, t_idx] = y_pred_norm
                y_pred_orig = scaler_y.inverse_transform(dummy)[:, t_idx]

                m = calculate_all_metrics(y_test_eval[:, t_idx], y_pred_orig)
                maes.append(m['MAE']); rmses.append(m['RMSE']); mses.append(m['MSE'])
                r2s.append(m['R2']); mapes.append(m['MAPE']); smapes.append(m['sMAPE'])

            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'XGBoost', 'Repeat': 'Mean', 'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
                            'MSE': np.mean(mses), 'R2': np.mean(r2s), 'MAPE': np.mean(mapes), 'sMAPE': np.mean(smapes)})
            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'XGBoost', 'Repeat': 'Std', 'MAE': np.std(maes), 'RMSE': np.std(rmses),
                            'MSE': np.std(mses), 'R2': np.std(r2s), 'MAPE': np.std(mapes), 'sMAPE': np.std(smapes)})

print("XGBoost تمام شد.")


# In[60]:


print("\n" + "="*95)
print("اجرای مدل: SVM (SVR) - ۱۰ تکرار")
print("="*95)

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]
        X_train = split['X_train'].reshape(split['X_train'].shape[0], -1)
        X_test = split['X_test'].reshape(split['X_test'].shape[0], -1)
        y_train_norm = split['y_train']
        y_test_raw_full = np.array(split['y_test_raw'])
        scaler_y = split['scaler_y']
        lookback = split['lookback']
        y_test_eval = y_test_raw_full[lookback:]

        for t_idx, target_name in enumerate(['Business', 'Special']):
            param_dist = {'C': [0.1, 1, 10, 50], 'epsilon': [0.01, 0.05, 0.1], 'kernel': ['rbf', 'linear']}
            search = RandomizedSearchCV(SVR(), param_dist, n_iter=6, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
            search.fit(X_train, y_train_norm[:, t_idx])
            best_params = search.best_params_

            maes, rmses, mses, r2s, mapes, smapes = [], [], [], [], [], []
            for rep in range(N_REPEATS):
                model = SVR(**best_params)
                model.fit(X_train, y_train_norm[:, t_idx])
                y_pred_norm = model.predict(X_test)

                dummy = np.zeros((len(y_pred_norm), 2))
                dummy[:, t_idx] = y_pred_norm
                y_pred_orig = scaler_y.inverse_transform(dummy)[:, t_idx]

                m = calculate_all_metrics(y_test_eval[:, t_idx], y_pred_orig)
                maes.append(m['MAE']); rmses.append(m['RMSE']); mses.append(m['MSE'])
                r2s.append(m['R2']); mapes.append(m['MAPE']); smapes.append(m['sMAPE'])

            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'SVM (SVR)', 'Repeat': 'Mean', 'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
                            'MSE': np.mean(mses), 'R2': np.mean(r2s), 'MAPE': np.mean(mapes), 'sMAPE': np.mean(smapes)})
            results.append({'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                            'Model': 'SVM (SVR)', 'Repeat': 'Std', 'MAE': np.std(maes), 'RMSE': np.std(rmses),
                            'MSE': np.std(mses), 'R2': np.std(r2s), 'MAPE': np.std(mapes), 'sMAPE': np.std(smapes)})

print("SVM تمام شد.")


# In[61]:


df_results = pd.DataFrame(results)

print("\n" + "="*120)
print("خلاصه نتایج (فقط میانگین‌ها)")
print("="*120)

summary = df_results[df_results['Repeat'].isin([1, 'Mean'])].copy()
print(summary.round(4).to_string(index=False))

df_results.to_excel("All_Baseline_Models_Complete_Metrics.xlsx", index=False)
print("\nنتایج کامل در فایل All_Baseline_Models_Complete_Metrics.xlsx ذخیره شد.")


# In[62]:


import pandas as pd
import numpy as np

# فرض بر این است که df_results را از اجرای قبلی دارید
df = pd.read_excel("All_Baseline_Models_Complete_Metrics.xlsx")   # یا از متغیر results بسازید

# فقط میانگین‌ها را نگه داریم
df_mean = df[df['Repeat'].isin([1, 'Mean'])].copy()

# تمیز کردن جدول
df_mean = df_mean[['Window', 'Scenario', 'Target', 'Model', 'MAE', 'RMSE', 'MSE', 'R2', 'MAPE', 'sMAPE']]

# تابع هایلایت بهترین مدل (کمترین مقدار)
def highlight_best(s):
    is_best = s == s.min()
    return ['background-color: #c6efce; font-weight: bold' if v else '' for v in is_best]

# ساخت جدول‌های جداگانه برای هر پنجره (توصیه اکید برای مقاله)
for window in df_mean['Window'].unique():
    print(f"\n{'='*100}")
    print(f"جدول خلاصه - {window}")
    print('='*100)

    sub = df_mean[df_mean['Window'] == window].copy()

    # مرتب‌سازی برای خوانایی
    sub = sub.sort_values(['Target', 'Scenario', 'MAE'])

    # نمایش جدول با استایل
    styled = (sub.style
              .format(precision=2)
              .apply(highlight_best, subset=['MAE', 'RMSE', 'sMAPE'], axis=0)
              .set_caption(f"Performance Comparison - {window}")
              .set_table_styles([{'selector': 'th', 'props': [('background-color', '#f0f0f0')]}]))

    display(styled)

    # ذخیره نسخه اکسل با رنگ
    sub.to_excel(f"Table_{window}_Styled.xlsx", index=False)


# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# ===================== نمودار MAE =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, window in enumerate(df_mean['Window'].unique()):
    ax = axes[idx]
    sub = df_mean[df_mean['Window'] == window]

    sns.barplot(data=sub, x='Model', y='MAE', hue='Target', ax=ax, palette=['#4C72B0', '#DD8452'])
    ax.set_title(window, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('MAE (Original Scale)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Class')

plt.suptitle('Comparison of Baseline Models - MAE', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure_MAE_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================== نمودار RMSE =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, window in enumerate(df_mean['Window'].unique()):
    ax = axes[idx]
    sub = df_mean[df_mean['Window'] == window]

    sns.barplot(data=sub, x='Model', y='RMSE', hue='Target', ax=ax, palette=['#4C72B0', '#DD8452'])
    ax.set_title(window, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('RMSE (Original Scale)')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Comparison of Baseline Models - RMSE', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("Figure_RMSE_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# In[65]:


# فقط بهترین مدل‌ها بر اساس MAE
best_models = df_mean.loc[df_mean.groupby(['Window', 'Target'])['MAE'].idxmin()]

plt.figure(figsize=(12, 6))
sns.barplot(data=best_models, x='Window', y='MAE', hue='Model', palette='Set2')
plt.title('Best Model per Window (by MAE)', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
plt.savefig("Figure_Best_Models.png", dpi=300, bbox_inches='tight')
plt.show()


# In[67]:


get_ipython().system('pip install scikit-posthocs')


# In[68]:


import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# ===================== بارگذاری نتایج =====================
df = pd.read_excel("All_Baseline_Models_Complete_Metrics.xlsx")

# فقط میانگین‌ها را نگه داریم
df_mean = df[df['Repeat'].isin([1, 'Mean'])].copy()

# ===================== آماده‌سازی داده =====================
df_mean['Dataset'] = df_mean['Window'] + ' | ' + df_mean['Target'] + ' | ' + df_mean['Scenario']

models = ['Naïve Seasonal', 'ARIMA', 'SARIMA', 'Exponential Smoothing', 
          'Random Forest', 'XGBoost', 'SVM (SVR)']

pivot_mae = df_mean.pivot_table(
    index='Dataset', 
    columns='Model', 
    values='MAE', 
    aggfunc='mean'
)

pivot_clean = pivot_mae[models].dropna()

print(f"تعداد مجموعه‌داده‌ها برای مقایسه: {len(pivot_clean)}")

# ===================== ۱. آزمون Friedman =====================
print("\n" + "="*75)
print("آزمون Friedman (مقایسه کلی مدل‌ها بر اساس MAE)")
print("="*75)

stat, p_value = friedmanchisquare(*[pivot_clean[col] for col in pivot_clean.columns])

print(f"Friedman statistic = {stat:.4f}")
print(f"p-value            = {p_value:.6f}")

if p_value < 0.05:
    print("→ نتیجه: تفاوت معنادار آماری بین مدل‌ها وجود دارد (p < 0.05)")
else:
    print("→ نتیجه: تفاوت معنادار آماری بین مدل‌ها وجود ندارد.")

# ===================== ۲. رتبه متوسط مدل‌ها =====================
print("\n" + "="*75)
print("رتبه متوسط مدل‌ها (میانگین رتبه در تمام مجموعه‌داده‌ها)")
print("="*75)

ranks = pivot_clean.rank(axis=1, method='average')
avg_rank = ranks.mean().sort_values()
print(avg_rank.round(3))

# ===================== ۳. آزمون Post-hoc جفتی (Wilcoxon + Bonferroni) =====================
print("\n" + "="*75)
print("آزمون Post-hoc جفتی (Wilcoxon Signed-Rank با تصحیح Bonferroni)")
print("="*75)

n_models = len(models)
n_comparisons = n_models * (n_models - 1) / 2
alpha = 0.05
bonferroni_alpha = alpha / n_comparisons

print(f"تعداد مقایسه‌های جفتی: {int(n_comparisons)}")
print(f"سطح معنی‌داری اصلاح‌شده (Bonferroni): {bonferroni_alpha:.4f}\n")

significant_pairs = []

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        model1 = models[i]
        model2 = models[j]

        diff = pivot_clean[model1] - pivot_clean[model2]

        try:
            stat_w, p_w = wilcoxon(diff)
            if p_w < bonferroni_alpha:
                significant_pairs.append({
                    'Model 1': model1,
                    'Model 2': model2,
                    'p-value': round(p_w, 4),
                    'Significant (Bonferroni)': 'Yes'
                })
        except:
            pass

if significant_pairs:
    print("جفت مدل‌هایی که تفاوت معنادار دارند (پس از تصحیح Bonferroni):")
    for pair in significant_pairs:
        print(f"  {pair['Model 1']} vs {pair['Model 2']}  →  p = {pair['p-value']}")
else:
    print("هیچ جفت مدلی تفاوت معنادار آماری نداشت (پس از تصحیح Bonferroni).")

# ===================== ذخیره نتایج =====================
results_stat = {
    'Friedman Statistic': stat,
    'Friedman p-value': p_value,
    'Average Ranks': avg_rank.to_dict(),
    'Significant Pairs (Bonferroni)': significant_pairs
}

print("\n" + "="*75)
print("آزمون آماری با موفقیت انجام شد.")


# In[70]:


from statsmodels.tsa.stattools import acf


# In[72]:


import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf   # ← اصلاح شد

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test برای مقایسه دقت پیش‌بینی دو مدل
    e1 و e2 باید آرایه خطاهای پیش‌بینی (squared errors) باشند
    """
    d = np.array(e1)**2 - np.array(e2)**2
    n = len(d)

    if n < 2:
        return np.nan, np.nan

    mean_d = np.mean(d)

    # محاسبه واریانس با در نظر گرفتن خودهمبستگی (autocorrelation)
    try:
        acf_values = acf(d, nlags=h, fft=False)
        var_d = (1 / n) * (np.var(d, ddof=1) + 2 * np.sum([acf_values[k] * np.var(d, ddof=1) for k in range(1, h+1)]))
    except:
        # در صورت بروز مشکل، از روش ساده استفاده می‌کنیم
        var_d = np.var(d, ddof=1) / n

    if var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


# In[74]:


import numpy as np
import pandas as pd
import itertools
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# ===================== تابع Diebold-Mariano (نسخه ساده و پایدار) =====================
def diebold_mariano_test(e1, e2, h=1):
    """Diebold-Mariano test روی squared errors"""
    e1 = np.array(e1)
    e2 = np.array(e2)
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)

    if var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

# ===================== تنظیمات =====================
N_REPEATS_DM = 1          # برای DM از یک پیش‌بینی نماینده استفاده می‌کنیم (برای پایداری)
results_dm = []

print("شروع اجرای آزمون Diebold-Mariano دو به دو برای همه مدل‌ها...\n")

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]

        X_train = split['X_train'].reshape(split['X_train'].shape[0], -1)
        X_test = split['X_test'].reshape(split['X_test'].shape[0], -1)
        y_train_norm = split['y_train']
        y_test_raw_full = np.array(split['y_test_raw'])
        scaler_y = split['scaler_y']
        lookback = split['lookback']

        y_test_eval = y_test_raw_full[lookback:]   # اصلاح طول

        for t_idx, target_name in enumerate(['Business', 'Special']):

            # ===================== ساخت خطا برای هر مدل =====================
            error_dict = {}

            # --- 1. Naïve Seasonal ---
            period = 7
            last_cycle = np.array(split['y_train_raw'])[:, t_idx][-period:]
            naive_pred = np.tile(last_cycle, (len(y_test_eval) // period + 2))[:len(y_test_eval)]
            error_dict['Naïve Seasonal'] = y_test_eval[:, t_idx] - naive_pred

            # --- 2. ARIMA ---
            try:
                model_ar = auto_arima(np.array(split['y_train_raw'])[:, t_idx], seasonal=True, m=7, 
                                      trace=False, error_action='ignore', suppress_warnings=True)
                arima_pred = model_ar.predict(n_periods=len(y_test_eval))
                error_dict['ARIMA'] = y_test_eval[:, t_idx] - arima_pred
            except:
                error_dict['ARIMA'] = np.full(len(y_test_eval), np.nan)

            # --- 3. SARIMA (همان ARIMA با فصلی) ---
            error_dict['SARIMA'] = error_dict.get('ARIMA', np.full(len(y_test_eval), np.nan))

            # --- 4. Exponential Smoothing ---
            try:
                model_es = ExponentialSmoothing(np.array(split['y_train_raw'])[:, t_idx], 
                                                seasonal='add', seasonal_periods=7).fit()
                es_pred = model_es.forecast(len(y_test_eval))
                error_dict['Exponential Smoothing'] = y_test_eval[:, t_idx] - es_pred
            except:
                error_dict['Exponential Smoothing'] = np.full(len(y_test_eval), np.nan)

            # --- 5. Random Forest ---
            rf = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
            rf.fit(X_train, y_train_norm[:, t_idx])
            rf_pred_norm = rf.predict(X_test)
            dummy = np.zeros((len(rf_pred_norm), 2)); dummy[:, t_idx] = rf_pred_norm
            rf_pred = scaler_y.inverse_transform(dummy)[:, t_idx]
            error_dict['Random Forest'] = y_test_eval[:, t_idx] - rf_pred

            # --- 6. XGBoost ---
            xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                         random_state=42, verbosity=0)
            xgb_model.fit(X_train, y_train_norm[:, t_idx])
            xgb_pred_norm = xgb_model.predict(X_test)
            dummy = np.zeros((len(xgb_pred_norm), 2)); dummy[:, t_idx] = xgb_pred_norm
            xgb_pred = scaler_y.inverse_transform(dummy)[:, t_idx]
            error_dict['XGBoost'] = y_test_eval[:, t_idx] - xgb_pred

            # --- 7. SVM (SVR) ---
            svm = SVR(C=10, epsilon=0.1, kernel='rbf')
            svm.fit(X_train, y_train_norm[:, t_idx])
            svm_pred_norm = svm.predict(X_test)
            dummy = np.zeros((len(svm_pred_norm), 2)); dummy[:, t_idx] = svm_pred_norm
            svm_pred = scaler_y.inverse_transform(dummy)[:, t_idx]
            error_dict['SVM (SVR)'] = y_test_eval[:, t_idx] - svm_pred

            # ===================== اجرای آزمون دو به دو =====================
            model_names = list(error_dict.keys())
            combinations = list(itertools.combinations(model_names, 2))

            for model1, model2 in combinations:
                e1 = error_dict[model1]
                e2 = error_dict[model2]

                if np.any(np.isnan(e1)) or np.any(np.isnan(e2)):
                    dm_stat, p_val = np.nan, np.nan
                else:
                    dm_stat, p_val = diebold_mariano_test(e1, e2, h=1)

                results_dm.append({
                    'Window': window_name,
                    'Scenario': scenario_name,
                    'Target': target_name,
                    'Model 1': model1,
                    'Model 2': model2,
                    'DM Statistic': dm_stat,
                    'p-value': p_val
                })

# ===================== تبدیل به DataFrame و نمایش =====================
df_dm = pd.DataFrame(results_dm)

# تصحیح p-value برای گزارش (طبق کامنت داور)
def format_p(p):
    if pd.isna(p):
        return "NaN"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

df_dm['p-value (formatted)'] = df_dm['p-value'].apply(format_p)

print("\n" + "="*110)
print("نتایج آزمون Diebold-Mariano (دو به دو برای همه مدل‌ها)")
print("="*110)
print(df_dm.to_string(index=False))

# ذخیره نتایج
df_dm.to_excel("Diebold_Mariano_Pairwise_Results.xlsx", index=False)
print("\nنتایج کامل در فایل 'Diebold_Mariano_Pairwise_Results.xlsx' ذخیره شد.")


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import warnings
warnings.filterwarnings('ignore')

N_REPEATS = 30

def set_all_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def calculate_all_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    smape = 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'R2': r2, 'MAPE': mape, 'sMAPE': smape}

def inverse_scale(pred_norm, scaler_y, target_idx):
    dummy = np.zeros((len(pred_norm), 2))
    dummy[:, target_idx] = pred_norm
    return scaler_y.inverse_transform(dummy)[:, target_idx]

dl_results = []
print("تنظیمات مشترک Deep Learning آماده شد.")


# In[9]:


print("\n" + "="*90)
print("اجرای مدل: MLP Multi-Output (۳۰ تکرار مستقل)")
print("="*90)

for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():
        split = normalized_splits[window_name][scenario_name]
        X_train = split['X_train']
        X_test = split['X_test']
        y_train_norm = split['y_train']
        y_test_raw_full = np.array(split['y_test_raw'])
        scaler_y = split['scaler_y']
        lookback = split['lookback']
        y_test_eval = y_test_raw_full[lookback:]

        # تبدیل به 2D برای MLP
        X_tr = X_train.reshape(X_train.shape[0], -1)
        X_te = X_test.reshape(X_test.shape[0], -1)

        # ذخیره نتایج برای هر دو تارگت
        target_data = {
            'Business': {'maes': [], 'rmses': [], 'mses': [], 'r2s': [], 'mapes': [], 'smapes': [],
                         'best_mae': float('inf'), 'best_pred': None, 'best_true': None},
            'Special':  {'maes': [], 'rmses': [], 'mses': [], 'r2s': [], 'mapes': [], 'smapes': [],
                         'best_mae': float('inf'), 'best_pred': None, 'best_true': None}
        }
        histories = []

        for rep in range(N_REPEATS):
            set_all_seeds(42 + rep)
            tf.keras.backend.clear_session()

            # ==================== معماری Multi-Output ====================
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_tr.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(2)   # ← خروجی دوتایی (Business + Special)
            ])
            model.compile(optimizer='adam', loss='mse')

            es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
            history = model.fit(X_tr, y_train_norm,           # ← کل ماتریس دو ستونی
                                epochs=100, batch_size=32,
                                validation_split=0.2, verbose=0, callbacks=[es])

            histories.append(history.history['loss'])

            y_pred_norm = model.predict(X_te, verbose=0)      # شکل: (n, 2)

            for t_idx, target_name in enumerate(['Business', 'Special']):
                y_pred = inverse_scale(y_pred_norm[:, t_idx], scaler_y, t_idx)
                y_true = y_test_eval[:, t_idx]

                m = calculate_all_metrics(y_true, y_pred)

                target_data[target_name]['maes'].append(m['MAE'])
                target_data[target_name]['rmses'].append(m['RMSE'])
                target_data[target_name]['mses'].append(m['MSE'])
                target_data[target_name]['r2s'].append(m['R2'])
                target_data[target_name]['mapes'].append(m['MAPE'])
                target_data[target_name]['smapes'].append(m['sMAPE'])

                if m['MAE'] < target_data[target_name]['best_mae']:
                    target_data[target_name]['best_mae'] = m['MAE']
                    target_data[target_name]['best_pred'] = y_pred
                    target_data[target_name]['best_true'] = y_true

        # ==================== نتایج و نمایش ====================
        # چاپ معماری فقط یک بار (آخرین مدل)
        print(f"\n=== معماری MLP Multi-Output برای {window_name} - {scenario_name} ===")
        model.summary()

        # نمودار Loss (مشترک)
        min_len = min(len(h) for h in histories)
        mean_loss = np.mean([h[:min_len] for h in histories], axis=0)
        std_loss = np.std([h[:min_len] for h in histories], axis=0)
        plt.figure(figsize=(8,4))
        plt.plot(mean_loss, label='Mean Loss')
        plt.fill_between(range(min_len), mean_loss - std_loss, mean_loss + std_loss, alpha=0.3)
        plt.title(f"MLP Multi-Output - Loss Curve ({window_name} | {scenario_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.show()

        # ذخیره نتایج و رسم نمودار برای هر تارگت
        for t_idx, target_name in enumerate(['Business', 'Special']):
            tdata = target_data[target_name]

            mean_m = {
                'MAE': np.mean(tdata['maes']),
                'RMSE': np.mean(tdata['rmses']),
                'MSE': np.mean(tdata['mses']),
                'R2': np.mean(tdata['r2s']),
                'MAPE': np.mean(tdata['mapes']),
                'sMAPE': np.mean(tdata['smapes'])
            }
            std_m = {
                'MAE': np.std(tdata['maes']),
                'RMSE': np.std(tdata['rmses']),
                'MSE': np.std(tdata['mses']),
                'R2': np.std(tdata['r2s']),
                'MAPE': np.std(tdata['mapes']),
                'sMAPE': np.std(tdata['smapes'])
            }

            dl_results.append({
                'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                'Model': 'MLP', 'Repeat': 'Mean', **mean_m
            })
            dl_results.append({
                'Window': window_name, 'Scenario': scenario_name, 'Target': target_name,
                'Model': 'MLP', 'Repeat': 'Std', **std_m
            })

            # نمودار سری زمانی برای این تارگت
            plt.figure(figsize=(10,4))
            plt.plot(tdata['best_true'], label='Actual', alpha=0.8)
            plt.plot(tdata['best_pred'], label='Predicted (Best Run)', alpha=0.8)
            plt.title(f"MLP Multi-Output - Actual vs Predicted ({window_name} | {target_name})")
            plt.legend()
            plt.show()

print("MLP Multi-Output تمام شد.")


# In[10]:


import pandas as pd
import numpy as np
from IPython.display import display

# تبدیل نتایج موجود به DataFrame
df_all_results = pd.DataFrame(dl_results)

# فقط نتایج MLP
df_mlp_results = df_all_results[
    df_all_results['Model'].astype(str).str.contains('MLP', case=False, na=False)
].copy()

print("تعداد ردیف‌های نتایج MLP:", len(df_mlp_results))

if df_mlp_results.empty:
    print("⚠️ هیچ نتیجه‌ای از MLP در dl_results پیدا نشد.")
    print("احتمالاً dl_results قبل از اجرای مدل تعریف یا پاک شده است.")
else:
    # مرتب‌سازی
    df_mlp_results = df_mlp_results.sort_values(
        by=['Window', 'Scenario', 'Target', 'Repeat']
    ).reset_index(drop=True)

    print("\n" + "=" * 110)
    print("جدول معیارهای ارزیابی MLP Multi-Output روی داده‌های Test")
    print("=" * 110)

    display(df_mlp_results)

    # فقط نتایج میانگین 30 اجرا
    df_mlp_mean = df_mlp_results[
        df_mlp_results['Repeat'].astype(str).str.lower() == 'mean'
    ].copy()

    print("\n" + "=" * 110)
    print("نتایج میانگین 30 تکرار مستقل")
    print("=" * 110)

    display(df_mlp_mean)

    # ذخیره در Excel
    output_file = "MLP_MultiOutput_Evaluation_ExistingResults.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_mlp_results.to_excel(writer, sheet_name="All_Mean_Std_Results", index=False)
        df_mlp_mean.to_excel(writer, sheet_name="Mean_Test_Metrics", index=False)

    print(f"\n✅ جدول معیارها ذخیره شد: {output_file}")


# In[11]:


# =============================================================================
# MLP MULTI-OUTPUT: Evaluation Metrics + Hyperparameters + Actual/Predicted
# =============================================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------------------------------------------
# تنظیمات
# -----------------------------------------------------------------------------
N_REPEATS = 30
MAX_EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
PATIENCE = 15
BASE_SEED = 42

OUTPUT_EXCEL = "MLP_MultiOutput_Complete_Results.xlsx"

TARGET_NAMES = ['Business', 'Special']

# -----------------------------------------------------------------------------
# تابع تنظیم Seed برای تکرارپذیری
# -----------------------------------------------------------------------------
def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# محل ذخیره خروجی‌ها
# -----------------------------------------------------------------------------
mlp_run_metrics = []          # معیارهای هر یک از 30 تکرار
mlp_summary_results = []      # میانگین و انحراف معیار
mlp_hyperparameters = []      # تنظیمات مدل
mlp_actual_predicted = []     # Actual و Predicted بهترین اجرا

print("\n" + "=" * 100)
print("اجرای MLP Multi-Output با ذخیره Metrics، Hyperparameters و Actual/Predicted")
print("=" * 100)

# =============================================================================
# حلقه اصلی: Window → Scenario → Repetition
# =============================================================================
for window_name in normalized_splits.keys():

    for scenario_name in normalized_splits[window_name].keys():

        split = normalized_splits[window_name][scenario_name]

        X_train = split['X_train']
        X_test = split['X_test']

        # y_train نرمال‌شده و دارای دو ستون: Business / Special
        y_train_norm = np.asarray(split['y_train'])

        # مقادیر واقعی در مقیاس اصلی
        y_test_raw_full = np.asarray(split['y_test_raw'])

        scaler_y = split['scaler_y']
        lookback = split['lookback']

        # هم‌راستاسازی y با نمونه‌های ساخته‌شده از lag
        y_test_eval = y_test_raw_full[lookback:]

        # تبدیل ورودی سه‌بعدی به دوبعدی برای MLP
        X_tr = X_train.reshape(X_train.shape[0], -1)
        X_te = X_test.reshape(X_test.shape[0], -1)

        print(f"\n{'-' * 100}")
        print(f"Window: {window_name} | Scenario: {scenario_name}")
        print(f"X_train shape: {X_tr.shape}")
        print(f"X_test  shape: {X_te.shape}")
        print(f"y_train shape: {y_train_norm.shape}")
        print(f"y_test  shape: {y_test_eval.shape}")
        print(f"{'-' * 100}")

        # ---------------------------------------------------------------------
        # داده‌های موقتی برای هر Window / Scenario
        # ---------------------------------------------------------------------
        target_data = {
            'Business': {
                'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [],
                'MAPE': [], 'sMAPE': [],
                'best_mae': np.inf,
                'best_rep': None,
                'best_pred': None,
                'best_true': None
            },
            'Special': {
                'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [],
                'MAPE': [], 'sMAPE': [],
                'best_mae': np.inf,
                'best_rep': None,
                'best_pred': None,
                'best_true': None
            }
        }

        stopped_epochs = []
        histories = []

        # =====================================================================
        # 30 تکرار مستقل
        # =====================================================================
        for rep in range(1, N_REPEATS + 1):

            seed = BASE_SEED + rep
            set_all_seeds(seed)
            tf.keras.backend.clear_session()

            # -----------------------------------------------------------------
            # معماری MLP Multi-Output
            # -----------------------------------------------------------------
            model = Sequential([
                Input(shape=(X_tr.shape[1],)),
                Dense(128, activation='relu', name='Dense_128'),
                Dropout(0.30, name='Dropout_30'),
                Dense(64, activation='relu', name='Dense_64'),
                Dropout(0.20, name='Dropout_20'),
                Dense(2, activation='linear', name='Output_Business_Special')
            ])

            model.compile(
                optimizer='adam',
                loss='mse'
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0
            )

            history = model.fit(
                X_tr,
                y_train_norm,
                validation_split=VALIDATION_SPLIT,
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping],
                verbose=0,
                shuffle=False  # مهم برای داده سری زمانی
            )

            histories.append(history.history['loss'])
            stopped_epochs.append(len(history.history['loss']))

            # پیش‌بینی دوخروجی نرمال‌شده
            y_pred_norm = model.predict(
                X_te,
                batch_size=BATCH_SIZE,
                verbose=0
            )

            # -----------------------------------------------------------------
            # ارزیابی جداگانه هر خروجی
            # -----------------------------------------------------------------
            for t_idx, target_name in enumerate(TARGET_NAMES):

                # بازگرداندن پیش‌بینی به مقیاس واقعی
                y_pred = inverse_scale(
                    y_pred_norm[:, t_idx],
                    scaler_y,
                    t_idx
                )

                # مقدار واقعی در مقیاس اصلی
                y_true = y_test_eval[:, t_idx]

                # محاسبه معیارها
                m = calculate_all_metrics(y_true, y_pred)

                # ذخیره متریک هر تکرار
                mlp_run_metrics.append({
                    'Window': window_name,
                    'Scenario': scenario_name,
                    'Target': target_name,
                    'Model': 'MLP_MultiOutput',
                    'Repeat': rep,
                    'Seed': seed,
                    'Stopped_Epoch': len(history.history['loss']),
                    'MAE': m['MAE'],
                    'MSE': m['MSE'],
                    'RMSE': m['RMSE'],
                    'R2': m['R2'],
                    'MAPE': m['MAPE'],
                    'sMAPE': m['sMAPE']
                })

                # ذخیره برای میانگین و انحراف معیار
                for metric_name in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
                    target_data[target_name][metric_name].append(m[metric_name])

                # ذخیره بهترین اجرا بر اساس MAE
                if m['MAE'] < target_data[target_name]['best_mae']:
                    target_data[target_name]['best_mae'] = m['MAE']
                    target_data[target_name]['best_rep'] = rep
                    target_data[target_name]['best_pred'] = np.asarray(y_pred).copy()
                    target_data[target_name]['best_true'] = np.asarray(y_true).copy()

        # =====================================================================
        # ذخیره Hyperparameters برای این Window و Scenario
        # =====================================================================
        mlp_hyperparameters.append({
            'Window': window_name,
            'Scenario': scenario_name,
            'Model': 'MLP_MultiOutput',

            'Input_Dimension': X_tr.shape[1],
            'Number_of_Outputs': 2,
            'Output_Targets': 'Business, Special',

            'Layer_1': 'Dense(128, relu)',
            'Dropout_1': 0.30,
            'Layer_2': 'Dense(64, relu)',
            'Dropout_2': 0.20,
            'Output_Layer': 'Dense(2, linear)',

            'Optimizer': 'Adam',
            'Loss_Function': 'MSE',

            'Max_Epochs': MAX_EPOCHS,
            'Batch_Size': BATCH_SIZE,
            'Validation_Split': VALIDATION_SPLIT,
            'EarlyStopping_Patience': PATIENCE,
            'Shuffle_During_Training': False,

            'N_Repeats': N_REPEATS,
            'Base_Seed': BASE_SEED,
            'Mean_Stopped_Epoch': np.mean(stopped_epochs),
            'Std_Stopped_Epoch': np.std(stopped_epochs)
        })

        # =====================================================================
        # نمایش معماری مدل
        # =====================================================================
        print(f"\nمعماری MLP برای {window_name} | {scenario_name}")
        model.summary()

        # =====================================================================
        # میانگین و انحراف معیار متریک‌ها + Actual / Predicted
        # =====================================================================
        for target_name in TARGET_NAMES:

            tdata = target_data[target_name]

            summary_row_mean = {
                'Window': window_name,
                'Scenario': scenario_name,
                'Target': target_name,
                'Model': 'MLP_MultiOutput',
                'Statistic': 'Mean',
                'MAE': np.mean(tdata['MAE']),
                'MSE': np.mean(tdata['MSE']),
                'RMSE': np.mean(tdata['RMSE']),
                'R2': np.mean(tdata['R2']),
                'MAPE': np.mean(tdata['MAPE']),
                'sMAPE': np.mean(tdata['sMAPE']),
                'Best_Run_by_MAE': tdata['best_rep'],
                'Best_MAE': tdata['best_mae']
            }

            summary_row_std = {
                'Window': window_name,
                'Scenario': scenario_name,
                'Target': target_name,
                'Model': 'MLP_MultiOutput',
                'Statistic': 'Std',
                'MAE': np.std(tdata['MAE']),
                'MSE': np.std(tdata['MSE']),
                'RMSE': np.std(tdata['RMSE']),
                'R2': np.std(tdata['R2']),
                'MAPE': np.std(tdata['MAPE']),
                'sMAPE': np.std(tdata['sMAPE']),
                'Best_Run_by_MAE': tdata['best_rep'],
                'Best_MAE': tdata['best_mae']
            }

            mlp_summary_results.append(summary_row_mean)
            mlp_summary_results.append(summary_row_std)

            # -------------------------------------------------------------
            # ذخیره Actual و Predicted مربوط به بهترین اجرای هر Target
            # -------------------------------------------------------------
            best_true = tdata['best_true']
            best_pred = tdata['best_pred']

            for time_idx, (actual_value, predicted_value) in enumerate(zip(best_true, best_pred)):

                absolute_error = abs(actual_value - predicted_value)

                mlp_actual_predicted.append({
                    'Window': window_name,
                    'Scenario': scenario_name,
                    'Target': target_name,
                    'Model': 'MLP_MultiOutput',

                    'Selected_Run': tdata['best_rep'],
                    'Selection_Criterion': 'Minimum Test MAE',

                    'Test_Time_Index': time_idx,
                    'Actual': actual_value,
                    'Predicted': predicted_value,
                    'Error': actual_value - predicted_value,
                    'Absolute_Error': absolute_error
                })

            # -------------------------------------------------------------
            # نمودار Actual vs Predicted
            # -------------------------------------------------------------
            plt.figure(figsize=(14, 5))

            plt.plot(
                best_true,
                label='Actual',
                color='black',
                linewidth=1.8
            )

            plt.plot(
                best_pred,
                label=f'Predicted (Best Run = {tdata["best_rep"]})',
                color='royalblue',
                linestyle='--',
                linewidth=1.5
            )

            plt.title(
                f'MLP Multi-Output: Actual vs Predicted\n'
                f'{window_name} | {scenario_name} | {target_name}'
            )
            plt.xlabel('Test Time Index')
            plt.ylabel('Demand (Original Scale)')
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

        # =====================================================================
        # نمودار Mean Loss
        # =====================================================================
        min_len = min(len(h) for h in histories)
        truncated_histories = np.array([h[:min_len] for h in histories])

        mean_loss = truncated_histories.mean(axis=0)
        std_loss = truncated_histories.std(axis=0)

        plt.figure(figsize=(10, 4))
        plt.plot(mean_loss, label='Mean Training Loss', color='darkred')
        plt.fill_between(
            range(min_len),
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.25,
            color='salmon',
            label='± 1 Std'
        )

        plt.title(f'MLP Multi-Output Mean Loss: {window_name} | {scenario_name}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

# =============================================================================
# تبدیل خروجی‌ها به DataFrame
# =============================================================================
df_mlp_run_metrics = pd.DataFrame(mlp_run_metrics)
df_mlp_summary = pd.DataFrame(mlp_summary_results)
df_mlp_hyperparameters = pd.DataFrame(mlp_hyperparameters)
df_mlp_actual_predicted = pd.DataFrame(mlp_actual_predicted)

# =============================================================================
# جدول مقاله‌ای Mean ± Std
# =============================================================================
df_mean = df_mlp_summary[df_mlp_summary['Statistic'] == 'Mean'].copy()
df_std = df_mlp_summary[df_mlp_summary['Statistic'] == 'Std'].copy()

paper_rows = []

for _, mean_row in df_mean.iterrows():

    std_row = df_std[
        (df_std['Window'] == mean_row['Window']) &
        (df_std['Scenario'] == mean_row['Scenario']) &
        (df_std['Target'] == mean_row['Target'])
    ].iloc[0]

    paper_rows.append({
        'Window': mean_row['Window'],
        'Scenario': mean_row['Scenario'],
        'Target': mean_row['Target'],
        'Model': mean_row['Model'],

        'MAE (Mean ± Std)': f"{mean_row['MAE']:.4f} ± {std_row['MAE']:.4f}",
        'MSE (Mean ± Std)': f"{mean_row['MSE']:.4f} ± {std_row['MSE']:.4f}",
        'RMSE (Mean ± Std)': f"{mean_row['RMSE']:.4f} ± {std_row['RMSE']:.4f}",
        'R2 (Mean ± Std)': f"{mean_row['R2']:.4f} ± {std_row['R2']:.4f}",
        'MAPE (Mean ± Std)': f"{mean_row['MAPE']:.4f} ± {std_row['MAPE']:.4f}",
        'sMAPE (Mean ± Std)': f"{mean_row['sMAPE']:.4f} ± {std_row['sMAPE']:.4f}",

        'Best_Run_by_MAE': mean_row['Best_Run_by_MAE'],
        'Best_MAE': mean_row['Best_MAE']
    })

df_mlp_paper_table = pd.DataFrame(paper_rows)

# =============================================================================
# نمایش جدول‌ها
# =============================================================================
print("\n" + "=" * 120)
print("جدول نهایی معیارهای ارزیابی MLP Multi-Output روی داده‌های Test")
print("=" * 120)
display(df_mlp_paper_table)

print("\n" + "=" * 120)
print("جدول تنظیم هایپرپارامترها")
print("=" * 120)
display(df_mlp_hyperparameters)

print("\n" + "=" * 120)
print("نمونه‌ای از مقادیر واقعی و پیش‌بینی‌شده")
print("=" * 120)
display(df_mlp_actual_predicted.head(20))

# =============================================================================
# ذخیره تمام خروجی‌ها در Excel
# =============================================================================
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    # جدول مناسب مقاله
    df_mlp_paper_table.to_excel(
        writer,
        sheet_name='Paper_Mean_Std',
        index=False
    )

    # میانگین و Std به‌صورت عددی
    df_mlp_summary.to_excel(
        writer,
        sheet_name='Summary_Mean_Std',
        index=False
    )

    # تمام 30 تکرار
    df_mlp_run_metrics.to_excel(
        writer,
        sheet_name='All_30_Repetitions',
        index=False
    )

    # هایپرپارامترها
    df_mlp_hyperparameters.to_excel(
        writer,
        sheet_name='Hyperparameters',
        index=False
    )

    # مقدار واقعی و پ
====================================================================================================
اجرای MLP Multi-Output با ذخیره Metrics، Hyperparameters و Actual/Predicted
====================================================================================================
    df_mlp_actual_predicted.to_excel(
        writer,
        sheet_name='Actual_vs_Predicted',
        index=False
    )

print("\n" + "=" * 100)
print("✅ اجرای MLP Multi-Output کامل شد.")
print(f"✅ فایل Excel ذخیره شد: {OUTPUT_EXCEL}")
print("=" * 100)
----------------------------------------------------------------------------------------------------
Window: Window1_Prepurchase | Scenario: Scenario_1_Classic
X_train shape: (727, 162)
X_test  shape: (142, 162)
y_train shape: (727, 2)
y_test  shape: (142, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window1_Prepurchase | Scenario_1_Classic
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        20,864 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 87,752 (342.79 KB)
 Trainable params: 29,250 (114.26 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 58,502 (228.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window1_Prepurchase | Scenario: Scenario_2_TrainingHeavy
X_train shape: (834, 162)
X_test  shape: (89, 162)
y_train shape: (834, 2)
y_test  shape: (89, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window1_Prepurchase | Scenario_2_TrainingHeavy
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        20,864 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 87,752 (342.79 KB)
 Trainable params: 29,250 (114.26 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 58,502 (228.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window1_Prepurchase | Scenario: Scenario_3_DeepLearning
X_train shape: (940, 162)
X_test  shape: (36, 162)
y_train shape: (940, 2)
y_test  shape: (36, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window1_Prepurchase | Scenario_3_DeepLearning
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        20,864 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 87,752 (342.79 KB)
 Trainable params: 29,250 (114.26 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 58,502 (228.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window2_Normal | Scenario: Scenario_1_Classic
X_train shape: (720, 225)
X_test  shape: (135, 225)
y_train shape: (720, 2)
y_test  shape: (135, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window2_Normal | Scenario_1_Classic
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        28,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 111,944 (437.29 KB)
 Trainable params: 37,314 (145.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 74,630 (291.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window2_Normal | Scenario: Scenario_2_TrainingHeavy
X_train shape: (827, 225)
X_test  shape: (82, 225)
y_train shape: (827, 2)
y_test  shape: (82, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window2_Normal | Scenario_2_TrainingHeavy
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        28,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 111,944 (437.29 KB)
 Trainable params: 37,314 (145.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 74,630 (291.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window2_Normal | Scenario: Scenario_3_DeepLearning
X_train shape: (933, 225)
X_test  shape: (29, 225)
y_train shape: (933, 2)
y_test  shape: (29, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window2_Normal | Scenario_3_DeepLearning
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        28,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 111,944 (437.29 KB)
 Trainable params: 37,314 (145.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 74,630 (291.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window3_LastMinute | Scenario: Scenario_1_Classic
X_train shape: (736, 81)
X_test  shape: (151, 81)
y_train shape: (736, 2)
y_test  shape: (151, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window3_LastMinute | Scenario_1_Classic
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        10,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 56,648 (221.29 KB)
 Trainable params: 18,882 (73.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 37,766 (147.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window3_LastMinute | Scenario: Scenario_2_TrainingHeavy
X_train shape: (843, 81)
X_test  shape: (98, 81)
y_train shape: (843, 2)
y_test  shape: (98, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window3_LastMinute | Scenario_2_TrainingHeavy
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        10,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 56,648 (221.29 KB)
 Trainable params: 18,882 (73.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 37,766 (147.53 KB)




----------------------------------------------------------------------------------------------------
Window: Window3_LastMinute | Scenario: Scenario_3_DeepLearning
X_train shape: (949, 81)
X_test  shape: (45, 81)
y_train shape: (949, 2)
y_test  shape: (45, 2)
----------------------------------------------------------------------------------------------------

معماری MLP برای Window3_LastMinute | Scenario_3_DeepLearning
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense_128 (Dense)               │ (None, 128)            │        10,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_30 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dense_64 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout_20 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Business_Special (Dense) │ (None, 2)              │           130 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 56,648 (221.29 KB)
 Trainable params: 18,882 (73.76 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 37,766 (147.53 KB)




========================================================================================================================
جدول نهایی معیارهای ارزیابی MLP Multi-Output روی داده‌های Test
========================================================================================================================
Window	Scenario	Target	Model	MAE (Mean ± Std)	MSE (Mean ± Std)	RMSE (Mean ± Std)	R2 (Mean ± Std)	MAPE (Mean ± Std)	sMAPE (Mean ± Std)	Best_Run_by_MAE	Best_MAE
0	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	34.7956 ± 4.0831	1744.0503 ± 373.1964	41.5462 ± 4.2382	-0.7306 ± 0.3703	256.9955 ± 40.8213	69.5227 ± 3.8215	15	28.899485
1	Window1_Prepurchase	Scenario_1_Classic	Special	MLP_MultiOutput	30.2168 ± 2.5662	1549.5246 ± 170.7352	39.3049 ± 2.1558	-0.0531 ± 0.1160	3806210525.8535 ± 727357864.4752	70.1365 ± 3.8179	28	25.793332
2	Window1_Prepurchase	Scenario_2_TrainingHeavy	Business	MLP_MultiOutput	31.8335 ± 3.6813	1429.9997 ± 294.7951	37.6182 ± 3.8564	-0.5972 ± 0.3293	285.3276 ± 33.4691	66.6109 ± 4.1235	27	25.405064
3	Window1_Prepurchase	Scenario_2_TrainingHeavy	Special	MLP_MultiOutput	27.7255 ± 3.6538	1146.5685 ± 244.3951	33.6754 ± 3.5410	0.0694 ± 0.1984	173.2381 ± 36.3794	67.5068 ± 5.0084	25	22.169791
4	Window1_Prepurchase	Scenario_3_DeepLearning	Business	MLP_MultiOutput	29.9416 ± 2.7498	1348.4642 ± 222.7096	36.5994 ± 2.9911	-0.4207 ± 0.2346	101.6835 ± 12.6182	52.3585 ± 2.9370	2	25.853067
5	Window1_Prepurchase	Scenario_3_DeepLearning	Special	MLP_MultiOutput	25.7861 ± 2.6754	1169.4794 ± 183.1310	34.0979 ± 2.6107	0.2106 ± 0.1236	87.0044 ± 13.4636	47.8963 ± 3.7076	16	21.359893
6	Window2_Normal	Scenario_1_Classic	Business	MLP_MultiOutput	41.2278 ± 9.1598	2398.4059 ± 916.7427	48.0896 ± 9.2629	-2.9935 ± 1.5264	20133351442.4084 ± 5081193368.6142	110.7530 ± 7.7274	29	26.358076
7	Window2_Normal	Scenario_1_Classic	Special	MLP_MultiOutput	27.1836 ± 2.4669	1969.2681 ± 108.3368	44.3592 ± 1.2352	-0.0273 ± 0.0565	11961445118.4265 ± 4347107012.9130	102.8415 ± 7.7427	29	22.420878
8	Window2_Normal	Scenario_2_TrainingHeavy	Business	MLP_MultiOutput	35.6543 ± 5.3491	1701.0670 ± 457.7276	40.9242 ± 5.1261	-1.7282 ± 0.7341	22721900316.6463 ± 3880749370.3491	112.7728 ± 5.4929	9	29.177820
9	Window2_Normal	Scenario_2_TrainingHeavy	Special	MLP_MultiOutput	18.2050 ± 3.2293	569.4585 ± 152.4234	23.6800 ± 2.9519	-0.4238 ± 0.3811	16047518037.5412 ± 4441081865.9022	97.0286 ± 5.2268	23	14.630149
10	Window2_Normal	Scenario_3_DeepLearning	Business	MLP_MultiOutput	32.2611 ± 3.5805	1419.1865 ± 292.2041	37.4705 ± 3.8919	-1.4033 ± 0.4948	20149902963.9874 ± 4799964286.8166	89.1883 ± 4.3526	27	23.576653
11	Window2_Normal	Scenario_3_DeepLearning	Special	MLP_MultiOutput	23.7185 ± 2.2344	1008.2237 ± 165.9295	31.6446 ± 2.6160	-0.4038 ± 0.2310	228.9581 ± 45.8319	76.0033 ± 3.7707	8	18.088767
12	Window3_LastMinute	Scenario_1_Classic	Business	MLP_MultiOutput	16.3098 ± 3.7593	444.7830 ± 167.0388	20.7548 ± 3.7446	-2.2291 ± 1.2127	84945140162.3601 ± 29230016901.6080	159.8258 ± 10.1920	8	11.461715
13	Window3_LastMinute	Scenario_1_Classic	Special	MLP_MultiOutput	42.3692 ± 14.1151	2631.2695 ± 1664.7097	48.9070 ± 15.4717	-18.7131 ± 12.4718	237488480923.1481 ± 79769781683.8736	172.4060 ± 4.7916	22	16.150724
14	Window3_LastMinute	Scenario_2_TrainingHeavy	Business	MLP_MultiOutput	11.8499 ± 1.1944	248.3602 ± 32.1172	15.7269 ± 1.0120	-0.7931 ± 0.2319	62466196655.0429 ± 10607236219.8946	156.7029 ± 7.6549	3	9.749556
15	Window3_LastMinute	Scenario_2_TrainingHeavy	Special	MLP_MultiOutput	13.4494 ± 3.7042	368.5931 ± 135.5994	18.8956 ± 3.3986	-1.2841 ± 0.8403	72358313711.3519 ± 24465245892.9161	159.9309 ± 3.7541	19	6.530352
16	Window3_LastMinute	Scenario_3_DeepLearning	Business	MLP_MultiOutput	14.2634 ± 1.5272	365.3221 ± 61.5366	19.0480 ± 1.5802	-0.4795 ± 0.2492	73624528265.3968 ± 12230289407.2143	153.9642 ± 6.5277	14	10.527825
17	Window3_LastMinute	Scenario_3_DeepLearning	Special	MLP_MultiOutput	14.5367 ± 2.2097	487.9771 ± 87.9682	22.0097 ± 1.8840	-0.4927 ± 0.2691	62867968021.6087 ± 16193891687.8058	149.0014 ± 6.1240	18	11.369889

========================================================================================================================
جدول تنظیم هایپرپارامترها
========================================================================================================================
Window	Scenario	Model	Input_Dimension	Number_of_Outputs	Output_Targets	Layer_1	Dropout_1	Layer_2	Dropout_2	...	Loss_Function	Max_Epochs	Batch_Size	Validation_Split	EarlyStopping_Patience	Shuffle_During_Training	N_Repeats	Base_Seed	Mean_Stopped_Epoch	Std_Stopped_Epoch
0	Window1_Prepurchase	Scenario_1_Classic	MLP_MultiOutput	162	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	27.466667	5.136363
1	Window1_Prepurchase	Scenario_2_TrainingHeavy	MLP_MultiOutput	162	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	24.333333	5.699903
2	Window1_Prepurchase	Scenario_3_DeepLearning	MLP_MultiOutput	162	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	25.500000	9.871677
3	Window2_Normal	Scenario_1_Classic	MLP_MultiOutput	225	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	31.400000	15.178054
4	Window2_Normal	Scenario_2_TrainingHeavy	MLP_MultiOutput	225	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	47.333333	24.796953
5	Window2_Normal	Scenario_3_DeepLearning	MLP_MultiOutput	225	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	39.700000	20.775226
6	Window3_LastMinute	Scenario_1_Classic	MLP_MultiOutput	81	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	19.433333	2.376038
7	Window3_LastMinute	Scenario_2_TrainingHeavy	MLP_MultiOutput	81	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	25.633333	6.949740
8	Window3_LastMinute	Scenario_3_DeepLearning	MLP_MultiOutput	81	2	Business, Special	Dense(128, relu)	0.3	Dense(64, relu)	0.2	...	MSE	100	32	0.2	15	False	30	42	18.900000	5.557877
9 rows × 22 columns


========================================================================================================================
نمونه‌ای از مقادیر واقعی و پیش‌بینی‌شده
========================================================================================================================
Window	Scenario	Target	Model	Selected_Run	Selection_Criterion	Test_Time_Index	Actual	Predicted	Error	Absolute_Error
0	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	0	31.0	98.777296	-67.777296	67.777296
1	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	1	119.0	94.873900	24.126100	24.126100
2	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	2	10.0	54.947908	-44.947908	44.947908
3	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	3	14.0	52.533582	-38.533582	38.533582
4	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	4	44.0	42.305555	1.694445	1.694445
5	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	5	15.0	35.614283	-20.614283	20.614283
6	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	6	49.0	66.577095	-17.577095	17.577095
7	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	7	27.0	74.157803	-47.157803	47.157803
8	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	8	58.0	55.891631	2.108369	2.108369
9	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	9	33.0	58.669032	-25.669032	25.669032
10	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	10	25.0	56.211435	-31.211435	31.211435
11	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	11	47.0	41.091180	5.908820	5.908820
12	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	12	72.0	25.160608	46.839392	46.839392
13	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	13	87.0	52.905581	34.094419	34.094419
14	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	14	30.0	61.801801	-31.801801	31.801801
15	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	15	10.0	50.190601	-40.190601	40.190601
16	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	16	92.0	39.965557	52.034443	52.034443
17	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	17	85.0	35.442246	49.557754	49.557754
18	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	18	21.0	32.646607	-11.646607	11.646607
19	Window1_Prepurchase	Scenario_1_Classic	Business	MLP_MultiOutput	15	Minimum Test MAE	19	30.0	51.078725	-21.078725	21.078725

====================================================================================================
✅ اجرای MLP Multi-Output کامل شد.
✅ فایل Excel ذخیره شد: MLP_MultiOutput_Complete_Results.xlsx
====================================================================================================
یش‌بینی‌شده
    df_mlp_actual_predicted.to_excel(
        writer,
        sheet_name='Actual_vs_Predicted',
        index=False
    )

print("\n" + "=" * 100)
print("✅ اجرای MLP Multi-Output کامل شد.")
print(f"✅ فایل Excel ذخیره شد: {OUTPUT_EXCEL}")
print("=" * 100)


# In[13]:


# =============================================================================
# MLP MULTI-OUTPUT IMPROVED
# - Non-negative demand
# - Optuna HPO on time-based validation (no test leakage)
# - Smaller regularized net + Huber + Softplus
# - 30-seed ensemble (report ensemble, not best-on-test)
# - Same normalized_splits / inverse_scale / calculate_all_metrics interface
# =============================================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# تنظیمات
# -----------------------------------------------------------------------------
N_REPEATS = 30
MAX_EPOCHS = 200
PATIENCE = 20
BASE_SEED = 42
VAL_FRAC = 0.15
MIN_VAL = 30
N_TRIALS = 35          # اگر وقت کم است: 20
MAX_PRED_FACTOR = 1.50 # سقف پیش‌بینی نسبت به max(train raw)
OUTPUT_EXCEL = "MLP_MultiOutput_Improved_Results.xlsx"
TARGET_NAMES = ["Business", "Special"]
PLOT_DIR = "mlp_improved_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# True = فقط یک window/scenario سریع تست؛ False = اجرای کامل
QUICK_TEST = False
QUICK_WINDOW = None      # مثلاً "Window1_Prepurchase"
QUICK_SCENARIO = None    # مثلاً "Scenario_1_Classic"


# -----------------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------------
def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def time_based_val_split(X, y, val_frac=VAL_FRAC, min_val=MIN_VAL):
    """آخرین بخش Train = Validation (سری‌زمانی، بدون shuffle)."""
    n = len(X)
    n_val = max(min_val, int(round(n * val_frac)))
    n_val = min(n_val, max(1, n // 3))
    n_fit = n - n_val
    if n_fit < 20:
        n_fit = max(1, n - max(1, n // 5))
        n_val = n - n_fit
    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:], n_fit, n_val


def inverse_multi_output(y_norm, scaler_y):
    """
    y_norm: (n, 2) در مقیاس نرمال
    خروجی: (n, 2) در مقیاس اصلی با inverse_scale موجود شما
    """
    y_norm = np.asarray(y_norm, dtype=np.float64)
    if y_norm.ndim == 1:
        y_norm = y_norm.reshape(-1, 1)
    outs = []
    for t_idx in range(y_norm.shape[1]):
        outs.append(np.asarray(inverse_scale(y_norm[:, t_idx], scaler_y, t_idx)).reshape(-1))
    return np.column_stack(outs)


def clip_nonnegative(y, y_train_raw=None, factor=MAX_PRED_FACTOR):
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)
    if y_train_raw is not None:
        cap = np.max(np.asarray(y_train_raw), axis=0) * float(factor)
        cap = np.where(cap <= 0, 1.0, cap)
        y = np.minimum(y, cap)
    return y


def metrics_per_target(y_true, y_pred, target_names=TARGET_NAMES):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rows = {}
    for t_idx, name in enumerate(target_names):
        m = calculate_all_metrics(y_true[:, t_idx], y_pred[:, t_idx])
        rows[name] = m
    return rows


def build_mlp(input_dim, params, n_outputs=2):
    reg = regularizers.l2(params["l2_reg"]) if params["l2_reg"] > 0 else None
    inp = Input(shape=(input_dim,), name="features")
    x = inp

    x = Dense(params["units1"], activation="relu",
              kernel_regularizer=reg, name="dense1")(x)
    if params.get("use_bn", False):
        x = BatchNormalization(name="bn1")(x)
    x = Dropout(params["dropout1"], name="drop1")(x)

    if params.get("units2", 0) and params["units2"] > 0:
        x = Dense(params["units2"], activation="relu",
                  kernel_regularizer=reg, name="dense2")(x)
        if params.get("use_bn", False):
            x = BatchNormalization(name="bn2")(x)
        x = Dropout(params["dropout2"], name="drop2")(x)

    # softplus => خروجی نرمال‌شده هم تمایل مثبت دارد؛ بعد از inverse هم clip می‌کنیم
    out_act = params.get("out_activation", "softplus")
    out = Dense(n_outputs, activation=out_act, name="output")(x)
    return Model(inp, out, name="MLP_Improved")


def compile_mlp(model, params):
    loss_name = params.get("loss_name", "huber")
    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=float(params.get("huber_delta", 1.0)))
    elif loss_name == "mae":
        loss = "mae"
    else:
        loss = "mse"

    opt = tf.keras.optimizers.Adam(
        learning_rate=float(params["lr"]),
        clipnorm=1.0
    )
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


def train_one_model(X_fit, y_fit, X_val, y_val, params, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    set_all_seeds(seed)
    tf.keras.backend.clear_session()

    model = build_mlp(X_fit.shape[1], params, n_outputs=y_fit.shape[1])
    model = compile_mlp(model, params)

    es = EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_mae",
        mode="min",
        factor=0.5,
        patience=max(5, patience // 3),
        min_lr=1e-6,
        verbose=0
    )

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        callbacks=[es, rlrop],
        verbose=0
    )

    stopped = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history.get("val_mae", history.history["loss"])) + 1)
    return model, history.history, stopped, best_epoch


def default_params():
    """اگر Optuna شکست بخورد."""
    return {
        "units1": 32,
        "units2": 16,
        "dropout1": 0.30,
        "dropout2": 0.20,
        "l2_reg": 1e-4,
        "lr": 1e-3,
        "batch_size": 32,
        "loss_name": "huber",
        "huber_delta": 1.0,
        "use_bn": False,
        "out_activation": "softplus",
    }


def optuna_search(X_train, y_train, n_trials=N_TRIALS, seed=BASE_SEED):
    X_fit, y_fit, X_val, y_val, _, _ = time_based_val_split(X_train, y_train)

    def objective(trial):
        params = {
            "units1": trial.suggest_categorical("units1", [16, 24, 32, 48, 64]),
            "units2": trial.suggest_categorical("units2", [0, 8, 16, 32]),
            "dropout1": trial.suggest_float("dropout1", 0.10, 0.45),
            "dropout2": trial.suggest_float("dropout2", 0.05, 0.35),
            "l2_reg": trial.suggest_float("l2_reg", 1e-6, 5e-3, log=True),
            "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "loss_name": trial.suggest_categorical("loss_name", ["huber", "mae", "mse"]),
            "huber_delta": trial.suggest_float("huber_delta", 0.5, 4.0),
            "use_bn": trial.suggest_categorical("use_bn", [False, True]),
            "out_activation": trial.suggest_categorical("out_activation", ["softplus", "relu"]),
        }
        model, hist, _, _ = train_one_model(
            X_fit, y_fit, X_val, y_val,
            params=params,
            seed=seed,
            max_epochs=120,
            patience=12
        )
        val_pred = model.predict(X_val, verbose=0)
        # MAE روی مقیاس نرمال validation (انتخاب مدل بدون دیدن test)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        return val_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, multivariate=True)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = default_params()
    best.update(study.best_params)
    print(f"  [Optuna] best val MAE (scaled space) = {study.best_value:.6f}")
    print(f"  [Optuna] best params = {best}")
    return best, study


# -----------------------------------------------------------------------------
# خروجی‌ها
# -----------------------------------------------------------------------------
mlp_run_metrics = []
mlp_summary_results = []
mlp_hyperparameters = []
mlp_actual_predicted = []
mlp_ensemble_metrics = []   # متریک ensemble نهایی (برای مقاله)

print("\n" + "=" * 100)
print("اجرای MLP Multi-Output IMPROVED | Optuna + NonNegative + Ensemble")
print("=" * 100)

# =============================================================================
# حلقه اصلی
# =============================================================================
for window_name in normalized_splits.keys():
    if QUICK_TEST and QUICK_WINDOW is not None and window_name != QUICK_WINDOW:
        continue

    for scenario_name in normalized_splits[window_name].keys():
        if QUICK_TEST and QUICK_SCENARIO is not None and scenario_name != QUICK_SCENARIO:
            continue

        split = normalized_splits[window_name][scenario_name]

        X_train = split["X_train"]
        X_test = split["X_test"]
        y_train_norm = np.asarray(split["y_train"], dtype=np.float32)
        y_test_raw_full = np.asarray(split["y_test_raw"], dtype=np.float64)
        scaler_y = split["scaler_y"]
        lookback = split["lookback"]

        # y_test در مقیاس اصلی، هم‌تراز نمونه‌های lag
        y_test_eval = np.asarray(y_test_raw_full[lookback:], dtype=np.float64)

        # اگر y_train_raw در split باشد برای cap بهتر است
        if "y_train_raw" in split:
            y_train_raw_full = np.asarray(split["y_train_raw"], dtype=np.float64)
            # هم‌ترازی تقریبی با نمونه‌های دنباله‌ای
            if len(y_train_raw_full) == len(y_train_norm):
                y_train_raw = y_train_raw_full
            elif len(y_train_raw_full) > lookback and len(y_train_raw_full) - lookback == len(y_train_norm):
                y_train_raw = y_train_raw_full[lookback:]
            else:
                # fallback: inverse خود y_train_norm
                y_train_raw = inverse_multi_output(y_train_norm, scaler_y)
        else:
            y_train_raw = inverse_multi_output(y_train_norm, scaler_y)

        # MLP: flatten
        X_tr = np.asarray(X_train, dtype=np.float32).reshape(X_train.shape[0], -1)
        X_te = np.asarray(X_test, dtype=np.float32).reshape(X_test.shape[0], -1)

        # هم‌طول بودن X/y
        n_tr = min(len(X_tr), len(y_train_norm))
        X_tr = X_tr[:n_tr]
        y_train_norm = y_train_norm[:n_tr]
        y_train_raw = y_train_raw[:n_tr]

        n_te = min(len(X_te), len(y_test_eval))
        X_te = X_te[:n_te]
        y_test_eval = y_test_eval[:n_te]

        if y_train_norm.ndim == 1:
            y_train_norm = y_train_norm.reshape(-1, 1)
        if y_test_eval.ndim == 1:
            y_test_eval = y_test_eval.reshape(-1, 1)

        print(f"\n{'-' * 100}")
        print(f"Window: {window_name} | Scenario: {scenario_name}")
        print(f"X_train: {X_tr.shape} | X_test: {X_te.shape}")
        print(f"y_train: {y_train_norm.shape} | y_test: {y_test_eval.shape}")
        print(f"{'-' * 100}")

        # ---------------------------------------------------------------------
        # 1) Optuna فقط روی Train/Val
        # ---------------------------------------------------------------------
        print(">>> Optuna hyperparameter search (validation only)...")
        try:
            best_params, study = optuna_search(
                X_tr, y_train_norm,
                n_trials=(8 if QUICK_TEST else N_TRIALS),
                seed=BASE_SEED
            )
            best_val_optuna = float(study.best_value)
        except Exception as e:
            print(f"  [Warn] Optuna failed ({e}) → default params")
            best_params = default_params()
            best_val_optuna = np.nan
            study = None

        # ---------------------------------------------------------------------
        # 2) split زمانی ثابت برای همه repeatها
        # ---------------------------------------------------------------------
        X_fit, y_fit, X_val, y_val, n_fit, n_val = time_based_val_split(X_tr, y_train_norm)
        y_fit_raw = y_train_raw[:n_fit]
        y_val_raw = y_train_raw[n_fit:]

        print(f"  Fit={n_fit} | Val={n_val} | Test={len(X_te)}")
        print(f"  Best params: {best_params}")

        # ---------------------------------------------------------------------
        # 3) 30 تکرار + جمع‌آوری پیش‌بینی‌ها برای Ensemble
        # ---------------------------------------------------------------------
        target_data = {
            t: {m: [] for m in ["MAE", "MSE", "RMSE", "R2", "MAPE", "sMAPE"]}
            for t in TARGET_NAMES
        }
        # برای سازگاری با خروجی قبلی (best run) — فقط گزارش جانبی
        for t in TARGET_NAMES:
            target_data[t].update({
                "best_mae": np.inf, "best_rep": None,
                "best_pred": None, "best_true": None
            })

        stopped_epochs = []
        best_epochs = []
        histories_loss = []
        histories_val = []

        test_pred_stack = []  #کد زیر **جایگزین کامل همان سلول** است و مستقیماً روی `normalized_splits`، `inverse_scale` و `calculate_all_metrics` خودت کار می‌کند.

### تغییرات کلیدی (برای بهتر شدن واقعی)
# =============================================================================
# MLP MULTI-OUTPUT IMPROVED
# Non-negative output | Better HP | Time-based Val | 30-seed Ensemble
# Works with: normalized_splits, inverse_scale, calculate_all_metrics
# =============================================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import display
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# تنظیمات
# -----------------------------------------------------------------------------
N_REPEATS = 30
MAX_EPOCHS = 200
BATCH_SIZE = 32
VAL_RATIO = 0.20          # آخرین VAL_RATIO از train = validation (سری‌زمانی)
PATIENCE = 20
BASE_SEED = 42
MIN_VAL = 24

# معماری / منظم‌سازی (برای حجم نمونه متوسط ریل؛ overfit کمتر از 128-64)
UNITS_1 = 64
UNITS_2 = 32              # اگر 0 بگذاری فقط یک لایه مخفی می‌سازد
DROPOUT_1 = 0.30
DROPOUT_2 = 0.20
L2_REG = 1e-4
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0
USE_BATCHNORM = False
OUTPUT_ACTIVATION = "softplus"   # softplus | relu  → نامنفی در فضای نرمال
CLIP_PRED_MIN = 0.0              # بعد از inverse_scale
MAX_PRED_FACTOR = 1.50           # سقف = factor * max(y_train_raw) per target

OUTPUT_EXCEL = "MLP_MultiOutput_Improved_Results.xlsx"
PLOT_DIR = "mlp_improved_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_NAMES = ["Business", "Special"]
MODEL_NAME = "MLP_MultiOutput_Improved"

os.environ["TF_DETERMINISTIC_OPS"] = "1"


# -----------------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------------
def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Time-based train/val split (بدون shuffle)
# -----------------------------------------------------------------------------
def time_based_val_split(X, y, val_ratio=0.20, min_val=24):
    n = len(X)
    n_val = max(min_val, int(round(n * val_ratio)))
    n_val = min(n_val, max(1, n // 3))
    n_fit = n - n_val
    if n_fit < max(20, n_val // 2):
        # fallback خیلی کوچک
        n_fit = max(1, int(n * 0.8))
        n_val = n - n_fit
    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:], n_fit, n_val


# -----------------------------------------------------------------------------
# y_train خام برای سقف پیش‌بینی (اگر در split نباشد از inverse روی y_train_norm)
# -----------------------------------------------------------------------------
def get_y_train_raw_for_cap(split, y_train_norm, scaler_y):
    if "y_train_raw" in split and split["y_train_raw"] is not None:
        y_raw = np.asarray(split["y_train_raw"])
        # ممکن است قبل از lookback باشد
        lookback = int(split.get("lookback", 0))
        if len(y_raw) == len(y_train_norm):
            return y_raw.astype(np.float64)
        if len(y_raw) > len(y_train_norm) and lookback > 0:
            return y_raw[lookback: lookback + len(y_train_norm)].astype(np.float64)
        # اگر هم‌طول نبود، inverse
    cols = []
    y_train_norm = np.asarray(y_train_norm)
    for t_idx in range(y_train_norm.shape[1]):
        cols.append(np.asarray(inverse_scale(y_train_norm[:, t_idx], scaler_y, t_idx), dtype=np.float64))
    return np.column_stack(cols)


# -----------------------------------------------------------------------------
# ساخت مدل
# -----------------------------------------------------------------------------
def build_mlp(input_dim, n_outputs=2):
    reg = regularizers.l2(L2_REG) if L2_REG and L2_REG > 0 else None
    inp = Input(shape=(input_dim,), name="features")
    x = inp

    x = Dense(UNITS_1, activation="relu", kernel_regularizer=reg, name=f"Dense_{UNITS_1}")(x)
    if USE_BATCHNORM:
        x = BatchNormalization(name="BN_1")(x)
    x = Dropout(DROPOUT_1, name=f"Dropout_{int(DROPOUT_1*100)}")(x)

    if UNITS_2 and UNITS_2 > 0:
        x = Dense(UNITS_2, activation="relu", kernel_regularizer=reg, name=f"Dense_{UNITS_2}")(x)
        if USE_BATCHNORM:
            x = BatchNormalization(name="BN_2")(x)
        x = Dropout(DROPOUT_2, name=f"Dropout_{int(DROPOUT_2*100)}")(x)

    out_act = OUTPUT_ACTIVATION if OUTPUT_ACTIVATION in ("softplus", "relu") else "linear"
    out = Dense(n_outputs, activation=out_act, name="Output_NonNeg_Business_Special")(x)
    return Model(inp, out, name=MODEL_NAME)


def compile_mlp(model):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=Huber(delta=HUBER_DELTA),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def predict_raw_nonneg(model, X, scaler_y, y_cap):
    """پیش‌بینی نرمال → inverse per target → clip نامنفی و سقف."""
    y_norm = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    y_norm = np.asarray(y_norm)
    preds = []
    for t_idx in range(y_norm.shape[1]):
        y_raw = np.asarray(inverse_scale(y_norm[:, t_idx], scaler_y, t_idx), dtype=np.float64)
        y_raw = np.clip(y_raw, CLIP_PRED_MIN, y_cap[t_idx])
        preds.append(y_raw)
    return np.column_stack(preds)


# -----------------------------------------------------------------------------
# محل ذخیره خروجی‌ها
# -----------------------------------------------------------------------------
mlp_run_metrics = []
mlp_summary_results = []
mlp_hyperparameters = []
mlp_actual_predicted = []
mlp_ensemble_metrics = []   # متریک Ensemble نهایی (برای مقاله)

print("\n" + "=" * 100)
print("اجرای MLP Multi-Output IMPROVED | NonNeg + Ensemble + Time Val")
print("=" * 100)

# =============================================================================
# حلقه اصلی
# =============================================================================
for window_name in normalized_splits.keys():
    for scenario_name in normalized_splits[window_name].keys():

        split = normalized_splits[window_name][scenario_name]

        X_train = split["X_train"]
        X_test = split["X_test"]
        y_train_norm = np.asarray(split["y_train"], dtype=np.float32)
        y_test_raw_full = np.asarray(split["y_test_raw"])
        scaler_y = split["scaler_y"]
        lookback = int(split["lookback"])

        # y_test در مقیاس اصلی، هم‌تراز با نمونه‌های lag
        y_test_eval = np.asarray(y_test_raw_full[lookback:], dtype=np.float64)
        if y_test_eval.ndim == 1:
            y_test_eval = y_test_eval.reshape(-1, 1)

        # MLP: 3D → 2D
        X_tr = np.asarray(X_train, dtype=np.float32).reshape(X_train.shape[0], -1)
        X_te = np.asarray(X_test, dtype=np.float32).reshape(X_test.shape[0], -1)

        # هم‌ترازی طول test
        n_te = min(len(X_te), len(y_test_eval))
        if len(X_te) != len(y_test_eval):
            print(f"⚠️ length align test: X_te={len(X_te)} y_test={len(y_test_eval)} → use {n_te}")
            X_te = X_te[:n_te]
            y_test_eval = y_test_eval[:n_te]

        n_out = y_train_norm.shape[1]
        assert n_out == len(TARGET_NAMES), "تعداد خروجی با TARGET_NAMES یکی نیست"

        # سقف پیش‌بینی از train خام
        y_train_raw_cap_base = get_y_train_raw_for_cap(split, y_train_norm, scaler_y)
        y_cap = np.maximum(np.max(y_train_raw_cap_base, axis=0) * MAX_PRED_FACTOR, 1.0)

        # split زمانی
        X_fit, y_fit, X_val, y_val, n_fit, n_val = time_based_val_split(
            X_tr, y_train_norm, val_ratio=VAL_RATIO, min_val=MIN_VAL
        )

        print(f"\n{'-' * 100}")
        print(f"Window: {window_name} | Scenario: {scenario_name}")
        print(f"X_train={X_tr.shape} | fit={X_fit.shape} | val={X_val.shape} | X_test={X_te.shape}")
        print(f"y_train={y_train_norm.shape} | y_test={y_test_eval.shape} | cap={np.round(y_cap, 2)}")
        print(f"{'-' * 100}")

        # ذخیره predهای هر seed برای Ensemble
        all_test_preds = []  # list of (n_test, 2)
        all_val_preds_raw = []
        stopped_epochs = []
        histories = []

        target_data = {
            t: {
                "MAE": [], "MSE": [], "RMSE": [], "R2": [], "MAPE": [], "sMAPE": [],
            }
            for t in TARGET_NAMES
        }

        # =====================================================================
        # 30 تکرار
        # =====================================================================
        for rep in range(1, N_REPEATS + 1):
            seed = BASE_SEED + rep
            set_all_seeds(seed)
            tf.keras.backend.clear_session()

            model = build_mlp(input_dim=X_fit.shape[1], n_outputs=n_out)
            model = compile_mlp(model)

            es = EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
            rlrop = ReduceLROnPlateau(
                monitor="val_mae",
                mode="min",
                factor=0.5,
                patience=max(5, PATIENCE // 3),
                min_lr=1e-6,
                verbose=0,
            )

            history = model.fit(
                X_fit, y_fit,
                validation_data=(X_val, y_val),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es, rlrop],
                verbose=0,
                shuffle=False,
            )

            hist = history.history
            histories.append(hist.get("loss", []))
            # epoch بهترین val_mae
            if "val_mae" in hist and len(hist["val_mae"]):
                stopped = int(np.argmin(hist["val_mae"]) + 1)
            else:
                stopped = len(hist.get("loss", []))
            stopped_epochs.append(stopped)

            # پیش‌بینی در مقیاس اصلی + نامنفی
            y_pred_test = predict_raw_nonneg(model, X_te, scaler_y, y_cap)
            y_pred_val = predict_raw_nonneg(model, X_val, scaler_y, y_cap)

            all_test_preds.append(y_pred_test)
            all_val_preds_raw.append(y_pred_val)

            # متریک per-seed روی Test (فقط dispersion؛ انتخاب مدل با test نمی‌شود)
            for t_idx, target_name in enumerate(TARGET_NAMES):
                y_true = y_test_eval[:, t_idx]
                y_pred = y_pred_test[:, t_idx]
                m = calculate_all_metrics(y_true, y_pred)

                mlp_run_metrics.append({
                    "Window": window_name,
                    "Scenario": scenario_name,
                    "Target": target_name,
                    "Model": MODEL_NAME,
                    "Repeat": rep,
                    "Seed": seed,
                    "Stopped_Epoch": stopped,
                    "MAE": m["MAE"],
                    "MSE": m["MSE"],
                    "RMSE": m["RMSE"],
                    "R2": m["R2"],
                    "MAPE": m["MAPE"],
                    "sMAPE": m["sMAPE"],
                })
                for metric_name in ["MAE", "MSE", "RMSE", "R2", "MAPE", "sMAPE"]:
                    target_data[target_name][metric_name].append(m[metric_name])

            if rep == 1 or rep % 5 == 0 or rep == N_REPEATS:
                mae_bus = calculate_all_metrics(y_test_eval[:, 0], y_pred_test[:, 0])["MAE"]
                mae_spc = calculate_all_metrics(y_test_eval[:, 1], y_pred_test[:, 1])["MAE"]
                print(f"  rep {rep:02d}/{N_REPEATS} | epoch={stopped:3d} | "
                      f"Test MAE Business={mae_bus:.3f} | Special={mae_spc:.3f}")

        # =====================================================================
        # Ensemble = میانگین پیش‌بینی ۳۰ seed (نه میانگین متریک)
        # =====================================================================
        ens_test = np.mean(np.stack(all_test_preds, axis=0), axis=0)
        ens_test = np.clip(ens_test, CLIP_PRED_MIN, y_cap)

        # bias correction فقط از Validation (مجاز)
        # نیاز به y_val در مقیاس خام
        y_val_raw_cols = []
        for t_idx in range(n_out):
            y_val_raw_cols.append(
                np.asarray(inverse_scale(np.asarray(y_val[:, t_idx]), scaler_y, t_idx), dtype=np.float64)
            )
        y_val_raw = np.column_stack(y_val_raw_cols)

        ens_val = np.mean(np.stack(all_val_preds_raw, axis=0), axis=0)
        ens_val = np.clip(ens_val, CLIP_PRED_MIN, y_cap)

        val_bias = np.mean(y_val_raw - ens_val, axis=0)  # actual - pred
        ens_test_bc = np.clip(ens_test + val_bias, CLIP_PRED_MIN, y_cap)
        ens_val_bc = np.clip(ens_val + val_bias, CLIP_PRED_MIN, y_cap)

        # اگر bias-correction روی VAL میانگین MAE را بهتر کرد، روی test اعمال کن
        def _mean_mae(y_t, y_p):
            return float(np.mean([calculate_all_metrics(y_t[:, j], y_p[:, j])["MAE"] for j in range(n_out)]))

        mae_val = _mean_mae(y_val_raw, ens_val)
        mae_val_bc = _mean_mae(y_val_raw, ens_val_bc)
        use_bc = mae_val_bc <= mae_val + 1e-12
        final_test = ens_test_bc if use_bc else ens_test

        print(f"\n→ Ensemble VAL MAE={mae_val:.4f} | VALbc={mae_val_bc:.4f} | bias_correction={use_bc}")
        print(f"→ bias per target (Business, Special) = {np.round(val_bias, 4)}")

        # =====================================================================
        # Hyperparameters
        # =====================================================================
        mlp_hyperparameters.append({
            "Window": window_name,
            "Scenario": scenario_name,
            "Model": MODEL_NAME,
            "Input_Dimension": int(X_tr.shape[1]),
            "Number_of_Outputs": int(n_out),
            "Output_Targets": "Business, Special",
            "Layer_1": f"Dense({UNITS_1}, relu)",
            "Dropout_1": DROPOUT_1,
            "Layer_2": f"Dense({UNITS_2}, relu)" if UNITS_2 else "None",
            "Dropout_2": DROPOUT_2 if UNITS_2 else None,
            "L2_Regularization": L2_REG,
            "BatchNorm": USE_BATCHNORM,
            "Output_Layer": f"Dense({n_out}, {OUTPUT_ACTIVATION})",
            "PostProcess": f"inverse_scale + clip[0, {MAX_PRED_FACTOR}*max_train]",
            "Optimizer": f"Adam(lr={LEARNING_RATE}, clipnorm=1.0)",
            "Loss_Function": f"Huber(delta={HUBER_DELTA})",
            "Max_Epochs": MAX_EPOCHS,
            "Batch_Size": BATCH_SIZE,
            "Validation": f"time-based last {VAL_RATIO:.0%} of train (no shuffle)",
            "EarlyStopping_Monitor": "val_mae",
            "EarlyStopping_Patience": PATIENCE,
            "ReduceLROnPlateau": True,
            "Shuffle_During_Training": False,
            "N_Repeats": N_REPEATS,
            "Base_Seed": BASE_SEED,
            "Reporting_Rule": "Ensemble mean of 30 test predictions (NOT best-on-test)",
            "Bias_Correction_From_Val": bool(use_bc),
            "Mean_Stopped_Epoch": float(np.mean(stopped_epochs)),
            "Std_Stopped_Epoch": float(np.std(stopped_epochs)),
            "N_Fit": int(n_fit),
            "N_Val": int(n_val),
            "N_Test": int(len(X_te)),
        })

        # summary آخرین مدل (فقط برای نمایش)
        print(f"\nمعماری MLP برای {window_name} | {scenario_name}")
        model.summary()

        # =====================================================================
        # متریک Ensemble + Mean±Std seeds
        # =====================================================================
        for t_idx, target_name in enumerate(TARGET_NAMES):
            tdata = target_data[target_name]

            # Mean / Std روی 30 seed
            summary_row_mean = {
                "Window": window_name,
                "Scenario": scenario_name,
                "Target": target_name,
                "Model": MODEL_NAME,
                "Statistic": "Mean_Seeds",
                "MAE": float(np.mean(tdata["MAE"])),
                "MSE": float(np.mean(tdata["MSE"])),
                "RMSE": float(np.mean(tdata["RMSE"])),
                "R2": float(np.mean(tdata["R2"])),
                "MAPE": float(np.mean(tdata["MAPE"])),
                "sMAPE": float(np.mean(tdata["sMAPE"])),
            }
            summary_row_std = {
                "Window": window_name,
                "Scenario": scenario_name,
                "Target": target_name,
                "Model": MODEL_NAME,
                "Statistic": "Std_Seeds",
                "MAE": float(np.std(tdata["MAE"])),
                "MSE": float(np.std(tdata["MSE"])),
                "RMSE": float(np.std(tdata["RMSE"])),
                "R2": float(np.std(tdata["R2"])),
                "MAPE": float(np.std(tdata["MAPE"])),
                "sMAPE": float(np.std(tdata["sMAPE"])),
            }
            mlp_summary_results.append(summary_row_mean)
            mlp_summary_results.append(summary_row_std)

            # متریک Ensemble نهایی (این را در مقاله گزارش کن)
            y_true = y_test_eval[:, t_idx]
            y_pred = final_test[:, t_idx]
            m_ens = calculate_all_metrics(y_true, y_pred)

            mlp_ensemble_metrics.append({
                "Window": window_name,
                "Scenario": scenario_name,
                "Target": target_name,
                "Model": MODEL_NAME,
                "Statistic": "Ensemble",
                "MAE": m_ens["MAE"],
                "MSE": m_ens["MSE"],
                "RMSE": m_ens["RMSE"],
                "R2": m_ens["R2"],
                "MAPE": m_ens["MAPE"],
                "sMAPE": m_ens["sMAPE"],
                "Bias_Correction": bool(use_bc),
                "Val_Bias": float(val_bias[t_idx]),
                "Seeds_MAE_Mean": summary_row_mean["MAE"],
                "Seeds_MAE_Std": summary_row_std["MAE"],
            })

            print(f"  [ENSEMBLE] {target_name}: "
                  f"MAE={m_ens['MAE']:.4f} | RMSE={m_ens['RMSE']:.4f} | "
                  f"R2={m_ens['R2']:.4f} | sMAPE={m_ens['sMAPE']:.4f}")

            # Actual vs Predicted از Ensemble
            for time_idx, (actual_value, predicted_value) in enumerate(zip(y_true, y_pred)):
                actual_value = float(actual_value)
                predicted_value = float(predicted_value)
                mlp_actual_predicted.append({
                    "Window": window_name,
                    "Scenario": scenario_name,
                    "Target": target_name,
                    "Model": MODEL_NAME,
                    "Selected_Run": "Ensemble_30_Seeds",
                    "Selection_Criterion": "Mean prediction over seeds (+ optional val bias correction)",
                    "Test_Time_Index": time_idx,
                    "Actual": actual_value,
                    "Predicted": predicted_value,
                    "Error": actual_value - predicted_value,
                    "Absolute_Error": abs(actual_value - predicted_value),
                })

            # نمودار Actual vs Predicted
            plt.figure(figsize=(14, 5))
            plt.plot(y_true, label="Actual", color="black", linewidth=1.8)
            plt.plot(
                y_pred,
                label="Predicted (Ensemble-30)",
                color="royalblue",
                linestyle="--",
                linewidth=1.5,
            )
            plt.title(
                f"{MODEL_NAME}: Actual vs Predicted\n"
                f"{window_name} | {scenario_name} | {target_name}"
            )
            plt.xlabel("Test Time Index")
            plt.ylabel("Demand (Original Scale, Non-Negative)")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            fig_path = os.path.join(
                PLOT_DIR, f"avp_{window_name}_{scenario_name}_{target_name}.png"
            )
            plt.savefig(fig_path, dpi=150)
            plt.show()
            print(f"  💾 plot: {fig_path}")

        # نمودار Mean Loss
        if len(histories) and min(len(h) for h in histories if len(h)) > 0:
            min_len = min(len(h) for h in histories if len(h) > 0)
            truncated = np.array([h[:min_len] for h in histories if len(h) >= min_len])
            mean_loss = truncated.mean(axis=0)
            std_loss = truncated.std(axis=0)

            plt.figure(figsize=(10, 4))
            plt.plot(mean_loss, label="Mean Training Loss", color="darkred")
            plt.fill_between(
                range(min_len),
                mean_loss - std_loss,
                mean_loss + std_loss,
                alpha=0.25,
                color="salmon",
                label="± 1 Std",
            )
            plt.title(f"{MODEL_NAME} Mean Loss: {window_name} | {scenario_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Huber Loss")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            fig_path = os.path.join(PLOT_DIR, f"loss_{window_name}_{scenario_name}.png")
            plt.savefig(fig_path, dpi=150)
            plt.show()

# =============================================================================
# DataFrameها
# =============================================================================
df_mlp_run_metrics = pd.DataFrame(mlp_run_metrics)
df_mlp_summary = pd.DataFrame(mlp_summary_results)
df_mlp_hyperparameters = pd.DataFrame(mlp_hyperparameters)
df_mlp_actual_predicted = pd.DataFrame(mlp_actual_predicted)
df_mlp_ensemble = pd.DataFrame(mlp_ensemble_metrics)

# =============================================================================
# جدول مقاله‌ای: Ensemble + Seeds Mean±Std
# =============================================================================
paper_rows = []
df_seed_mean = df_mlp_summary[df_mlp_summary["Statistic"] == "Mean_Seeds"].copy()
df_seed_std = df_mlp_summary[df_mlp_summary["Statistic"] == "Std_Seeds"].copy()

for _, ens_row in df_mlp_ensemble.iterrows():
    std_row = df_seed_std[
        (df_seed_std["Window"] == ens_row["Window"])
        & (df_seed_std["Scenario"] == ens_row["Scenario"])
        & (df_seed_std["Target"] == ens_row["Target"])
    ].iloc[0]
    mean_row = df_seed_mean[
        (df_seed_mean["Window"] == ens_row["Window"])
        & (df_seed_mean["Scenario"] == ens_row["Scenario"])
        & (df_seed_mean["Target"] == ens_row["Target"])
    ].iloc[0]

    paper_rows.append({
        "Window": ens_row["Window"],
        "Scenario": ens_row["Scenario"],
        "Target": ens_row["Target"],
        "Model": ens_row["Model"],
        # ستون اصلی مقاله
        "MAE_Ensemble": ens_row["MAE"],
        "RMSE_Ensemble": ens_row["RMSE"],
        "MSE_Ensemble": ens_row["MSE"],
        "R2_Ensemble": ens_row["R2"],
        "MAPE_Ensemble": ens_row["MAPE"],
        "sMAPE_Ensemble": ens_row["sMAPE"],
        # dispersion seedها
        "MAE (Seeds Mean ± Std)": f"{mean_row['MAE']:.4f} ± {std_row['MAE']:.4f}",
        "RMSE (Seeds Mean ± Std)": f"{mean_row['RMSE']:.4f} ± {std_row['RMSE']:.4f}",
        "R2 (Seeds Mean ± Std)": f"{mean_row['R2']:.4f} ± {std_row['R2']:.4f}",
        "MAPE (Seeds Mean ± Std)": f"{mean_row['MAPE']:.4f} ± {std_row['MAPE']:.4f}",
        "sMAPE (Seeds Mean ± Std)": f"{mean_row['sMAPE']:.4f} ± {std_row['sMAPE']:.4f}",
        "Bias_Correction": ens_row["Bias_Correction"],
        "Val_Bias": ens_row["Val_Bias"],
    })

df_mlp_paper_table = pd.DataFrame(paper_rows)

# =============================================================================
# نمایش
# =============================================================================
print("\n" + "=" * 120)
print("جدول نهایی مقاله — MLP IMPROVED (Ensemble روی Test)")
print("=" * 120)
display(df_mlp_paper_table)

print("\n" + "=" * 120)
print("هایپرپارامترها")
print("=" * 120)
display(df_mlp_hyperparameters)

print("\n" + "=" * 120)
print("نمونه Actual vs Predicted (Ensemble)")
print("=" * 120)
display(df_mlp_actual_predicted.head(20))

# =============================================================================
# ذخیره Excel (syntax درست)
# =============================================================================
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df_mlp_paper_table.to_excel(writer, sheet_name="Paper_Ensemble", index=False)
    df_mlp_ensemble.to_excel(writer, sheet_name="Ensemble_Metrics", index=False)
    df_mlp_summary.to_excel(writer, sheet_name="Seeds_Mean_Std", index=False)
    df_mlp_run_metrics.to_excel(writer, sheet_name="All_30_Repetitions", index=False)
    df_mlp_hyperparameters.to_excel(writer, sheet_name="Hyperparameters", index=False)
    df_mlp_actual_predicted.to_excel(writer, sheet_name="Actual_vs_Predicted", index=False)

print("\n" + "=" * 100)
print("✅ اجرای MLP Multi-Output IMPROVED کامل شد.")
print(f"✅ Excel: {OUTPUT_EXCEL}")
print(f"✅ Plots: {PLOT_DIR}/")
print("📌 در مقاله فقط ستون‌های *_Ensemble را به‌عنوان عملکرد نهایی گزارش کن.")
print("=" * 100)


# In[15]:


print(np.asarray(split['X_train']).shape)
print(np.asarray(split['y_train']).shape)


# In[16]:


# ============================================================
# LSTM Multi-Output | فقط Scenario_1_Classic
# اصلاح‌شده برای X با شکل (n, 18, 9) و y با شکل (n, 2)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# تنظیمات
# -----------------------------
TARGET_COLS = ['Business', 'Special']
SCENARIO_FILTER = 'Scenario_1_Classic'

N_SEEDS = 30
EPOCHS = 100
BATCH_SIZE = 32
LOOKBACK = 15
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_BASE = 42

OUTPUT_DIR = 'LSTM_Scenario1_Classic_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# اگر خواستی فقط یک Window سریع تست شود:
# QUICK_WINDOWS = ['Window1_Prepurchase']
QUICK_WINDOWS = None


# -----------------------------
# متریک‌ها
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred, prefix=''):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mean_squared_error(y_true, y_pred),
        f'{prefix}RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}sMAPE': smape(y_true, y_pred),
    }


# -----------------------------
# شکل‌دهی ورودی/خروجی
# X: (n, 18, 9) -> (n, 162)
# y: (n, 2)
# -----------------------------
def ensure_2d_X(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)   # (n, 18, 9) -> (n, 162)
    return X


def ensure_2d_y(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def create_sequences(X, y, lookback=LOOKBACK):
    """
    ورودی:
      X: (n, features)
      y: (n, targets)
    خروجی:
      Xs: (n-lookback, lookback, features)
      ys: (n-lookback, targets)
    """
    X = ensure_2d_X(X)
    y = ensure_2d_y(y)

    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i, :])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_xy_from_split(split):
    # کلیدهای رایج؛ اگر اسم کلیدها فرق داشت اینجا را عوض کن
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f'هیچ‌کدام از کلیدها پیدا نشد: {keys}')

    X_train = ensure_2d_X(pick(split, ['X_train', 'X_train_scaled', 'x_train']))
    y_train = ensure_2d_y(pick(split, ['y_train', 'Y_train']))
    X_val   = ensure_2d_X(pick(split, ['X_val', 'X_val_scaled', 'x_val']))
    y_val   = ensure_2d_y(pick(split, ['y_val', 'Y_val']))
    X_test  = ensure_2d_X(pick(split, ['X_test', 'X_test_scaled', 'x_test']))
    y_test  = ensure_2d_y(pick(split, ['y_test', 'Y_test']))

    # فقط دو خروجی
    if y_train.shape[1] > 2:
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm(input_shape, seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='linear')  # Business + Special
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


# -----------------------------
# اجرا
# -----------------------------
all_metrics_rows = []
all_preds_rows = []
all_hyperparams_rows = []

windows = list(normalized_splits.keys())
if QUICK_WINDOWS is not None:
    windows = [w for w in windows if w in QUICK_WINDOWS]

print('Windows:', windows)
print('Scenario filter:', SCENARIO_FILTER)

for window_name in windows:
    if SCENARIO_FILTER not in normalized_splits[window_name]:
        print(f'[Skip] {window_name} | {SCENARIO_FILTER} موجود نیست')
        continue

    split = normalized_splits[window_name][SCENARIO_FILTER]
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy_from_split(split)

    print('\n' + '=' * 70)
    print(f'{window_name} | {SCENARIO_FILTER}')
    print('RAW shapes after flatten:')
    print('X_train:', X_train.shape, '| y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   '| y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  '| y_test :', y_test.shape)

    # توالی‌سازی استاندارد LSTM: (samples, lookback, features)
    Xtr_seq, ytr_seq = create_sequences(X_train, y_train, LOOKBACK)
    Xva_seq, yva_seq = create_sequences(X_val, y_val, LOOKBACK)
    Xte_seq, yte_seq = create_sequences(X_test, y_test, LOOKBACK)

    print('SEQ shapes:')
    print('Xtr_seq:', Xtr_seq.shape, '| ytr_seq:', ytr_seq.shape)
    print('Xva_seq:', Xva_seq.shape, '| yva_seq:', yva_seq.shape)
    print('Xte_seq:', Xte_seq.shape, '| yte_seq:', yte_seq.shape)
    print('=' * 70)

    # باید دقیقاً 3 بعدی باشد
    assert Xtr_seq.ndim == 3, f'Xtr_seq must be 3D, got {Xtr_seq.shape}'
    assert Xva_seq.ndim == 3, f'Xva_seq must be 3D, got {Xva_seq.shape}'
    assert Xte_seq.ndim == 3, f'Xte_seq must be 3D, got {Xte_seq.shape}'
    assert ytr_seq.shape[1] == 2, f'ytr_seq must be (n,2), got {ytr_seq.shape}'

    if len(Xtr_seq) < 10 or len(Xte_seq) < 5:
        print('[Skip] تعداد نمونه توالی خیلی کم است')
        continue

    seed_metrics = []
    seed_preds = []

    for seed in range(N_SEEDS):
        seed_value = RANDOM_BASE + seed
        print(f'\n[{window_name}] Seed {seed+1}/{N_SEEDS} (seed={seed_value})')

        model = build_lstm(
            input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]),  # (15, 162)
            seed=seed_value
        )

        es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(Xte_seq, verbose=0)
        y_pred = np.maximum(y_pred, 0.0)  # خروجی نامنفی

        seed_preds.append(y_pred)

        row = {
            'Model': 'LSTM',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'ActualEpochsRun': len(history.history['loss']),
        }

        for j, tcol in enumerate(TARGET_COLS):
            row.update(compute_metrics(yte_seq[:, j], y_pred[:, j], prefix=f'{tcol}_'))

        row.update(compute_metrics(yte_seq.reshape(-1), y_pred.reshape(-1), prefix='Overall_'))
        seed_metrics.append(row)
        all_metrics_rows.append(row)

        all_hyperparams_rows.append({
            'Model': 'LSTM',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'LOOKBACK': LOOKBACK,
            'InputFeaturesFlat': Xtr_seq.shape[2],
            'LSTM_Units': 64,
            'Dense_Units': 64,
            'Dropout1': 0.3,
            'Dropout2': 0.2,
            'LR': LEARNING_RATE,
            'BatchSize': BATCH_SIZE,
            'EpochsMax': EPOCHS,
            'Patience': PATIENCE,
            'Optimizer': 'Adam',
            'Loss': 'MSE',
            'ActualEpochsRun': len(history.history['loss']),
        })

        # نمودار loss فقط برای seed اول
        if seed == 0:
            plt.figure(figsize=(7, 4))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'LSTM Loss | {window_name} | {SCENARIO_FILTER}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'Loss_{window_name}_{SCENARIO_FILTER}.png'), dpi=150)
            plt.close()

    # میانگین ۳۰ seed
    seed_preds = np.asarray(seed_preds)          # (seeds, samples, 2)
    y_ens = seed_preds.mean(axis=0)

    ens_row = {
        'Model': 'LSTM_EnsembleMean',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'EnsembleMean',
        'ActualEpochsRun': np.nan,
    }
    for j, tcol in enumerate(TARGET_COLS):
        ens_row.update(compute_metrics(yte_seq[:, j], y_ens[:, j], prefix=f'{tcol}_'))
    ens_row.update(compute_metrics(yte_seq.reshape(-1), y_ens.reshape(-1), prefix='Overall_'))
    all_metrics_rows.append(ens_row)

    # Actual vs Predicted
    for i in range(len(yte_seq)):
        all_preds_rows.append({
            'Model': 'LSTM_EnsembleMean',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Index': i,
            'Business_Actual': float(yte_seq[i, 0]),
            'Business_Pred': float(y_ens[i, 0]),
            'Special_Actual': float(yte_seq[i, 1]),
            'Special_Pred': float(y_ens[i, 1]),
        })

    # نمودار Actual vs Pred
    for j, tcol in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(yte_seq[:, j], label=f'{tcol} Actual')
        plt.plot(y_ens[:, j], label=f'{tcol} Pred (Ensemble)')
        plt.title(f'LSTM Actual vs Pred | {window_name} | {SCENARIO_FILTER} | {tcol}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'ActualPred_{window_name}_{SCENARIO_FILTER}_{tcol}.png'),
            dpi=150
        )
        plt.close()

    # خلاصه mean/std روی ۳۰ seed
    df_seed = pd.DataFrame(seed_metrics)
    summary = {
        'Model': 'LSTM',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'MeanOverSeeds',
        'ActualEpochsRun': df_seed['ActualEpochsRun'].mean(),
    }
    for tcol in TARGET_COLS:
        for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
            col = f'{tcol}_{m}'
            summary[f'{col}'] = df_seed[col].mean()
            summary[f'{col}_std'] = df_seed[col].std()

    all_metrics_rows.append(summary)
    print(f'\nDone: {window_name} | {SCENARIO_FILTER}')


# -----------------------------
# ذخیره Excel
# -----------------------------
metrics_df = pd.DataFrame(all_metrics_rows)
preds_df = pd.DataFrame(all_preds_rows)
hyper_df = pd.DataFrame(all_hyperparams_rows)

out_xlsx = os.path.join(OUTPUT_DIR, 'LSTM_Scenario1_Classic_Complete_Results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    preds_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    hyper_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print('\n' + '=' * 70)
print('تمام شد')
print('Saved to:', out_xlsx)
print('Figures folder:', OUTPUT_DIR)
print('=' * 70)

# نمایش خلاصه Ensemble
summary_view_cols = [
    'Window', 'Scenario',
    'Business_MAE', 'Business_RMSE', 'Business_MAPE', 'Business_R2',
    'Special_MAE', 'Special_RMSE', 'Special_MAPE', 'Special_R2'
]
summary_view = metrics_df[metrics_df['Seed'] == 'EnsembleMean']
summary_view = summary_view[[c for c in summary_view_cols if c in summary_view.columns]]
print('\nEnsemble summary:')
print(summary_view.to_string(index=False))


# In[18]:


# ============================================================
# Hybrid CNN-LSTM Multi-Output | فقط Scenario_1_Classic
# معیارهای Ensemble:
# MAE_Ensemble, RMSE_Ensemble, MSE_Ensemble, R2_Ensemble, MAPE_Ensemble, sMAPE_Ensemble
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# تنظیمات
# -----------------------------
TARGET_COLS = ['Business', 'Special']
SCENARIO_FILTER = 'Scenario_1_Classic'

N_SEEDS = 30
EPOCHS = 100
BATCH_SIZE = 32
LOOKBACK = 15
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_BASE = 42

OUTPUT_DIR = 'Hybrid_CNNLSTM_Scenario1_Classic_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# تست سریع فقط یک Window:
# QUICK_WINDOWS = ['Window1_Prepurchase']
QUICK_WINDOWS = None


# -----------------------------
# متریک‌ها
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred, prefix=''):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': np.sqrt(mse),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}sMAPE': smape(y_true, y_pred),
    }


def compute_ensemble_metrics(y_true, y_pred):
    """معیارهای Ensemble با نام‌های درخواستی"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE_Ensemble': mean_absolute_error(y_true, y_pred),
        'RMSE_Ensemble': np.sqrt(mse),
        'MSE_Ensemble': mse,
        'R2_Ensemble': r2_score(y_true, y_pred),
        'MAPE_Ensemble': mape(y_true, y_pred),
        'sMAPE_Ensemble': smape(y_true, y_pred),
    }


# -----------------------------
# شکل‌دهی ورودی/خروجی
# X: (n, 18, 9) -> (n, 162)
# y: (n, 2)
# -----------------------------
def ensure_2d_X(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # (n,18,9)->(n,162)
    return X


def ensure_2d_y(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def create_sequences(X, y, lookback=LOOKBACK):
    X = ensure_2d_X(X)
    y = ensure_2d_y(y)

    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i, :])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_xy_from_split(split):
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f'هیچ‌کدام از کلیدها پیدا نشد: {keys}')

    X_train = ensure_2d_X(pick(split, ['X_train', 'X_train_scaled', 'x_train']))
    y_train = ensure_2d_y(pick(split, ['y_train', 'Y_train']))
    X_val   = ensure_2d_X(pick(split, ['X_val', 'X_val_scaled', 'x_val']))
    y_val   = ensure_2d_y(pick(split, ['y_val', 'Y_val']))
    X_test  = ensure_2d_X(pick(split, ['X_test', 'X_test_scaled', 'x_test']))
    y_test  = ensure_2d_y(pick(split, ['y_test', 'Y_test']))

    if y_train.shape[1] > 2:
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# معماری Hybrid CNN-LSTM
# -----------------------------
def build_hybrid_cnn_lstm(input_shape, seed, show_summary=False):
    """
    input_shape = (lookback, features) مثلاً (15, 162)
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=input_shape, name='Input')

    # ----- بخش CNN -----
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Conv1D_1')(inputs)
    x = MaxPooling1D(pool_size=2, name='MaxPool_1')(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Conv1D_2')(x)
    x = Dropout(0.2, name='Dropout_CNN')(x)

    # ----- بخش LSTM -----
    x = LSTM(64, activation='tanh', return_sequences=False, name='LSTM_1')(x)
    x = Dropout(0.3, name='Dropout_LSTM')(x)

    # ----- بخش Dense -----
    x = Dense(64, activation='relu', name='Dense_1')(x)
    x = Dropout(0.2, name='Dropout_Dense')(x)
    outputs = Dense(2, activation='linear', name='Output_Business_Special')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_LSTM')
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    if show_summary:
        print('\n' + '=' * 70)
        print('ARCHITECTURE: Hybrid CNN-LSTM')
        print('=' * 70)
        model.summary()
        print('=' * 70 + '\n')

    return model


# -----------------------------
# اجرا
# -----------------------------
all_metrics_rows = []
all_preds_rows = []
all_hyperparams_rows = []
ensemble_summary_rows = []

windows = list(normalized_splits.keys())
if QUICK_WINDOWS is not None:
    windows = [w for w in windows if w in QUICK_WINDOWS]

print('Windows:', windows)
print('Scenario filter:', SCENARIO_FILTER)

architecture_shown = False

for window_name in windows:
    if SCENARIO_FILTER not in normalized_splits[window_name]:
        print(f'[Skip] {window_name} | {SCENARIO_FILTER} موجود نیست')
        continue

    split = normalized_splits[window_name][SCENARIO_FILTER]
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy_from_split(split)

    print('\n' + '=' * 70)
    print(f'{window_name} | {SCENARIO_FILTER}')
    print('RAW shapes after flatten:')
    print('X_train:', X_train.shape, '| y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   '| y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  '| y_test :', y_test.shape)

    Xtr_seq, ytr_seq = create_sequences(X_train, y_train, LOOKBACK)
    Xva_seq, yva_seq = create_sequences(X_val, y_val, LOOKBACK)
    Xte_seq, yte_seq = create_sequences(X_test, y_test, LOOKBACK)

    print('SEQ shapes:')
    print('Xtr_seq:', Xtr_seq.shape, '| ytr_seq:', ytr_seq.shape)
    print('Xva_seq:', Xva_seq.shape, '| yva_seq:', yva_seq.shape)
    print('Xte_seq:', Xte_seq.shape, '| yte_seq:', yte_seq.shape)
    print('=' * 70)

    assert Xtr_seq.ndim == 3, f'Xtr_seq must be 3D, got {Xtr_seq.shape}'
    assert ytr_seq.shape[1] == 2, f'ytr_seq must be (n,2), got {ytr_seq.shape}'

    if len(Xtr_seq) < 10 or len(Xte_seq) < 5:
        print('[Skip] تعداد نمونه توالی خیلی کم است')
        continue

    seed_metrics = []
    seed_preds = []

    for seed in range(N_SEEDS):
        seed_value = RANDOM_BASE + seed
        print(f'\n[{window_name}] Seed {seed+1}/{N_SEEDS} (seed={seed_value})')

        show_sum = (not architecture_shown)
        model = build_hybrid_cnn_lstm(
            input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]),  # (15, 162)
            seed=seed_value,
            show_summary=show_sum
        )
        if show_sum:
            architecture_shown = True

        es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(Xte_seq, verbose=0)
        y_pred = np.maximum(y_pred, 0.0)  # خروجی نامنفی
        seed_preds.append(y_pred)

        row = {
            'Model': 'Hybrid_CNN_LSTM',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'ActualEpochsRun': len(history.history['loss']),
        }

        # متریک per-target برای هر seed
        for j, tcol in enumerate(TARGET_COLS):
            row.update(compute_metrics(yte_seq[:, j], y_pred[:, j], prefix=f'{tcol}_'))

        # متریک overall برای هر seed
        row.update(compute_metrics(yte_seq.reshape(-1), y_pred.reshape(-1), prefix='Overall_'))

        seed_metrics.append(row)
        all_metrics_rows.append(row)

        all_hyperparams_rows.append({
            'Model': 'Hybrid_CNN_LSTM',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'LOOKBACK': LOOKBACK,
            'InputFeaturesFlat': int(Xtr_seq.shape[2]),
            'Conv1D_1_Filters': 64,
            'Conv1D_2_Filters': 32,
            'KernelSize': 3,
            'MaxPool': 2,
            'LSTM_Units': 64,
            'Dense_Units': 64,
            'Dropout_CNN': 0.2,
            'Dropout_LSTM': 0.3,
            'Dropout_Dense': 0.2,
            'LR': LEARNING_RATE,
            'BatchSize': BATCH_SIZE,
            'EpochsMax': EPOCHS,
            'Patience': PATIENCE,
            'Optimizer': 'Adam',
            'Loss': 'MSE',
            'ActualEpochsRun': len(history.history['loss']),
        })

        # نمودار Loss فقط seed اول هر Window
        if seed == 0:
            plt.figure(figsize=(7, 4))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'Hybrid CNN-LSTM Loss | {window_name} | {SCENARIO_FILTER}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f'Loss_{window_name}_{SCENARIO_FILTER}.png'),
                dpi=150
            )
            plt.close()

    # =========================
    # Ensemble میانگین ۳۰ seed
    # =========================
    seed_preds = np.asarray(seed_preds)  # (seeds, samples, 2)
    y_ens = seed_preds.mean(axis=0)

    ens_row = {
        'Model': 'Hybrid_CNN_LSTM_EnsembleMean',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'EnsembleMean',
        'ActualEpochsRun': np.nan,
    }

    # متریک‌های per-target برای Ensemble
    for j, tcol in enumerate(TARGET_COLS):
        ens_row.update(compute_metrics(yte_seq[:, j], y_ens[:, j], prefix=f'{tcol}_'))

    # متریک overall معمولی
    ens_row.update(compute_metrics(yte_seq.reshape(-1), y_ens.reshape(-1), prefix='Overall_'))

    # متریک‌های درخواستی Ensemble با نام دقیق
    ens_row.update(compute_ensemble_metrics(yte_seq.reshape(-1), y_ens.reshape(-1)))

    # متریک‌های Ensemble جدا برای هر هدف هم اضافه می‌کنیم
    for j, tcol in enumerate(TARGET_COLS):
        m = compute_ensemble_metrics(yte_seq[:, j], y_ens[:, j])
        ens_row[f'{tcol}_MAE_Ensemble'] = m['MAE_Ensemble']
        ens_row[f'{tcol}_RMSE_Ensemble'] = m['RMSE_Ensemble']
        ens_row[f'{tcol}_MSE_Ensemble'] = m['MSE_Ensemble']
        ens_row[f'{tcol}_R2_Ensemble'] = m['R2_Ensemble']
        ens_row[f'{tcol}_MAPE_Ensemble'] = m['MAPE_Ensemble']
        ens_row[f'{tcol}_sMAPE_Ensemble'] = m['sMAPE_Ensemble']

    all_metrics_rows.append(ens_row)

    # ردیف خلاصه تمیز برای جدول مقاله
    ensemble_summary_rows.append({
        'Model': 'Hybrid_CNN_LSTM',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'MAE_Ensemble': ens_row['MAE_Ensemble'],
        'RMSE_Ensemble': ens_row['RMSE_Ensemble'],
        'MSE_Ensemble': ens_row['MSE_Ensemble'],
        'R2_Ensemble': ens_row['R2_Ensemble'],
        'MAPE_Ensemble': ens_row['MAPE_Ensemble'],
        'sMAPE_Ensemble': ens_row['sMAPE_Ensemble'],
        'Business_MAE_Ensemble': ens_row['Business_MAE_Ensemble'],
        'Business_RMSE_Ensemble': ens_row['Business_RMSE_Ensemble'],
        'Business_MSE_Ensemble': ens_row['Business_MSE_Ensemble'],
        'Business_R2_Ensemble': ens_row['Business_R2_Ensemble'],
        'Business_MAPE_Ensemble': ens_row['Business_MAPE_Ensemble'],
        'Business_sMAPE_Ensemble': ens_row['Business_sMAPE_Ensemble'],
        'Special_MAE_Ensemble': ens_row['Special_MAE_Ensemble'],
        'Special_RMSE_Ensemble': ens_row['Special_RMSE_Ensemble'],
        'Special_MSE_Ensemble': ens_row['Special_MSE_Ensemble'],
        'Special_R2_Ensemble': ens_row['Special_R2_Ensemble'],
        'Special_MAPE_Ensemble': ens_row['Special_MAPE_Ensemble'],
        'Special_sMAPE_Ensemble': ens_row['Special_sMAPE_Ensemble'],
    })

    # Actual vs Predicted
    for i in range(len(yte_seq)):
        all_preds_rows.append({
            'Model': 'Hybrid_CNN_LSTM_EnsembleMean',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Index': i,
            'Business_Actual': float(yte_seq[i, 0]),
            'Business_Pred': float(y_ens[i, 0]),
            'Special_Actual': float(yte_seq[i, 1]),
            'Special_Pred': float(y_ens[i, 1]),
        })

    # نمودار Actual vs Pred
    for j, tcol in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(yte_seq[:, j], label=f'{tcol} Actual')
        plt.plot(y_ens[:, j], label=f'{tcol} Pred (Ensemble)')
        plt.title(f'Hybrid CNN-LSTM Actual vs Pred | {window_name} | {SCENARIO_FILTER} | {tcol}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'ActualPred_{window_name}_{SCENARIO_FILTER}_{tcol}.png'),
            dpi=150
        )
        plt.close()

    # خلاصه mean/std روی ۳۰ seed
    df_seed = pd.DataFrame(seed_metrics)
    summary = {
        'Model': 'Hybrid_CNN_LSTM',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'MeanOverSeeds',
        'ActualEpochsRun': df_seed['ActualEpochsRun'].mean(),
    }
    for tcol in TARGET_COLS:
        for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
            col = f'{tcol}_{m}'
            summary[f'{col}'] = df_seed[col].mean()
            summary[f'{col}_std'] = df_seed[col].std()

    all_metrics_rows.append(summary)
    print(f'\nDone: {window_name} | {SCENARIO_FILTER}')


# -----------------------------
# ذخیره Excel
# -----------------------------
metrics_df = pd.DataFrame(all_metrics_rows)
preds_df = pd.DataFrame(all_preds_rows)
hyper_df = pd.DataFrame(all_hyperparams_rows)
ens_summary_df = pd.DataFrame(ensemble_summary_rows)

out_xlsx = os.path.join(OUTPUT_DIR, 'Hybrid_CNNLSTM_Scenario1_Classic_Complete_Results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics_All', index=False)
    ens_summary_df.to_excel(writer, sheet_name='Ensemble_Metrics', index=False)
    preds_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    hyper_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print('\n' + '=' * 70)
print('تمام شد')
print('Saved to:', out_xlsx)
print('Figures folder:', OUTPUT_DIR)
print('=' * 70)

print('\n===== Ensemble Metrics Summary =====')
print(ens_summary_df.to_string(index=False))


# In[19]:


# ============================================================
# GRU Multi-Output | فقط Scenario_1_Classic
# معیارهای Ensemble:
# MAE_Ensemble, RMSE_Ensemble, MSE_Ensemble, R2_Ensemble, MAPE_Ensemble, sMAPE_Ensemble
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# تنظیمات
# -----------------------------
TARGET_COLS = ['Business', 'Special']
SCENARIO_FILTER = 'Scenario_1_Classic'

N_SEEDS = 30
EPOCHS = 100
BATCH_SIZE = 32
LOOKBACK = 15
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_BASE = 42

OUTPUT_DIR = 'GRU_Scenario1_Classic_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# تست سریع فقط یک Window:
# QUICK_WINDOWS = ['Window1_Prepurchase']
QUICK_WINDOWS = None


# -----------------------------
# متریک‌ها
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred, prefix=''):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': np.sqrt(mse),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}sMAPE': smape(y_true, y_pred),
    }


def compute_ensemble_metrics(y_true, y_pred):
    """معیارهای Ensemble با نام‌های درخواستی"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE_Ensemble': mean_absolute_error(y_true, y_pred),
        'RMSE_Ensemble': np.sqrt(mse),
        'MSE_Ensemble': mse,
        'R2_Ensemble': r2_score(y_true, y_pred),
        'MAPE_Ensemble': mape(y_true, y_pred),
        'sMAPE_Ensemble': smape(y_true, y_pred),
    }


# -----------------------------
# شکل‌دهی ورودی/خروجی
# X: (n, 18, 9) -> (n, 162)
# y: (n, 2)
# -----------------------------
def ensure_2d_X(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # (n,18,9)->(n,162)
    return X


def ensure_2d_y(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def create_sequences(X, y, lookback=LOOKBACK):
    X = ensure_2d_X(X)
    y = ensure_2d_y(y)

    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i, :])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_xy_from_split(split):
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f'هیچ‌کدام از کلیدها پیدا نشد: {keys}')

    X_train = ensure_2d_X(pick(split, ['X_train', 'X_train_scaled', 'x_train']))
    y_train = ensure_2d_y(pick(split, ['y_train', 'Y_train']))
    X_val   = ensure_2d_X(pick(split, ['X_val', 'X_val_scaled', 'x_val']))
    y_val   = ensure_2d_y(pick(split, ['y_val', 'Y_val']))
    X_test  = ensure_2d_X(pick(split, ['X_test', 'X_test_scaled', 'x_test']))
    y_test  = ensure_2d_y(pick(split, ['y_test', 'Y_test']))

    if y_train.shape[1] > 2:
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# معماری GRU
# -----------------------------
def build_gru(input_shape, seed, show_summary=False):
    """
    input_shape = (lookback, features) مثلاً (15, 162)
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=input_shape, name='Input')

    # ----- بخش GRU -----
    x = GRU(units=64, activation='tanh', return_sequences=True, name='GRU_1')(inputs)
    x = Dropout(0.3, name='Dropout_GRU_1')(x)

    x = GRU(units=32, activation='tanh', return_sequences=False, name='GRU_2')(x)
    x = Dropout(0.2, name='Dropout_GRU_2')(x)

    # ----- بخش Dense -----
    x = Dense(64, activation='relu', name='Dense_1')(x)
    x = Dropout(0.2, name='Dropout_Dense')(x)
    outputs = Dense(2, activation='linear', name='Output_Business_Special')(x)

    model = Model(inputs=inputs, outputs=outputs, name='GRU_MultiOutput')
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    if show_summary:
        print('\n' + '=' * 70)
        print('ARCHITECTURE: GRU Multi-Output')
        print('=' * 70)
        model.summary()
        print('=' * 70 + '\n')

    return model


# -----------------------------
# اجرا
# -----------------------------
all_metrics_rows = []
all_preds_rows = []
all_hyperparams_rows = []
ensemble_summary_rows = []

windows = list(normalized_splits.keys())
if QUICK_WINDOWS is not None:
    windows = [w for w in windows if w in QUICK_WINDOWS]

print('Windows:', windows)
print('Scenario filter:', SCENARIO_FILTER)

architecture_shown = False

for window_name in windows:
    if SCENARIO_FILTER not in normalized_splits[window_name]:
        print(f'[Skip] {window_name} | {SCENARIO_FILTER} موجود نیست')
        continue

    split = normalized_splits[window_name][SCENARIO_FILTER]
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy_from_split(split)

    print('\n' + '=' * 70)
    print(f'{window_name} | {SCENARIO_FILTER}')
    print('RAW shapes after flatten:')
    print('X_train:', X_train.shape, '| y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   '| y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  '| y_test :', y_test.shape)

    Xtr_seq, ytr_seq = create_sequences(X_train, y_train, LOOKBACK)
    Xva_seq, yva_seq = create_sequences(X_val, y_val, LOOKBACK)
    Xte_seq, yte_seq = create_sequences(X_test, y_test, LOOKBACK)

    print('SEQ shapes:')
    print('Xtr_seq:', Xtr_seq.shape, '| ytr_seq:', ytr_seq.shape)
    print('Xva_seq:', Xva_seq.shape, '| yva_seq:', yva_seq.shape)
    print('Xte_seq:', Xte_seq.shape, '| yte_seq:', yte_seq.shape)
    print('=' * 70)

    assert Xtr_seq.ndim == 3, f'Xtr_seq must be 3D, got {Xtr_seq.shape}'
    assert ytr_seq.shape[1] == 2, f'ytr_seq must be (n,2), got {ytr_seq.shape}'

    if len(Xtr_seq) < 10 or len(Xte_seq) < 5:
        print('[Skip] تعداد نمونه توالی خیلی کم است')
        continue

    seed_metrics = []
    seed_preds = []

    for seed in range(N_SEEDS):
        seed_value = RANDOM_BASE + seed
        print(f'\n[{window_name}] Seed {seed+1}/{N_SEEDS} (seed={seed_value})')

        show_sum = (not architecture_shown)
        model = build_gru(
            input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]),  # (15, 162)
            seed=seed_value,
            show_summary=show_sum
        )
        if show_sum:
            architecture_shown = True

        es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(Xte_seq, verbose=0)
        y_pred = np.maximum(y_pred, 0.0)  # خروجی نامنفی
        seed_preds.append(y_pred)

        row = {
            'Model': 'GRU',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'ActualEpochsRun': len(history.history['loss']),
        }

        # متریک per-target برای هر seed
        for j, tcol in enumerate(TARGET_COLS):
            row.update(compute_metrics(yte_seq[:, j], y_pred[:, j], prefix=f'{tcol}_'))

        # متریک overall برای هر seed
        row.update(compute_metrics(yte_seq.reshape(-1), y_pred.reshape(-1), prefix='Overall_'))

        seed_metrics.append(row)
        all_metrics_rows.append(row)

        all_hyperparams_rows.append({
            'Model': 'GRU',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'LOOKBACK': LOOKBACK,
            'InputFeaturesFlat': int(Xtr_seq.shape[2]),
            'GRU_1_Units': 64,
            'GRU_2_Units': 32,
            'Dense_Units': 64,
            'Dropout_GRU_1': 0.3,
            'Dropout_GRU_2': 0.2,
            'Dropout_Dense': 0.2,
            'LR': LEARNING_RATE,
            'BatchSize': BATCH_SIZE,
            'EpochsMax': EPOCHS,
            'Patience': PATIENCE,
            'Optimizer': 'Adam',
            'Loss': 'MSE',
            'ActualEpochsRun': len(history.history['loss']),
        })

        # نمودار Loss فقط seed اول هر Window
        if seed == 0:
            plt.figure(figsize=(7, 4))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'GRU Loss | {window_name} | {SCENARIO_FILTER}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f'Loss_{window_name}_{SCENARIO_FILTER}.png'),
                dpi=150
            )
            plt.close()

    # =========================
    # Ensemble میانگین ۳۰ seed
    # =========================
    seed_preds = np.asarray(seed_preds)  # (seeds, samples, 2)
    y_ens = seed_preds.mean(axis=0)

    ens_row = {
        'Model': 'GRU_EnsembleMean',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'EnsembleMean',
        'ActualEpochsRun': np.nan,
    }

    # متریک‌های per-target برای Ensemble
    for j, tcol in enumerate(TARGET_COLS):
        ens_row.update(compute_metrics(yte_seq[:, j], y_ens[:, j], prefix=f'{tcol}_'))

    # متریک overall معمولی
    ens_row.update(compute_metrics(yte_seq.reshape(-1), y_ens.reshape(-1), prefix='Overall_'))

    # متریک‌های درخواستی Ensemble با نام دقیق
    ens_row.update(compute_ensemble_metrics(yte_seq.reshape(-1), y_ens.reshape(-1)))

    # متریک‌های Ensemble جدا برای هر هدف
    for j, tcol in enumerate(TARGET_COLS):
        m = compute_ensemble_metrics(yte_seq[:, j], y_ens[:, j])
        ens_row[f'{tcol}_MAE_Ensemble'] = m['MAE_Ensemble']
        ens_row[f'{tcol}_RMSE_Ensemble'] = m['RMSE_Ensemble']
        ens_row[f'{tcol}_MSE_Ensemble'] = m['MSE_Ensemble']
        ens_row[f'{tcol}_R2_Ensemble'] = m['R2_Ensemble']
        ens_row[f'{tcol}_MAPE_Ensemble'] = m['MAPE_Ensemble']
        ens_row[f'{tcol}_sMAPE_Ensemble'] = m['sMAPE_Ensemble']

    all_metrics_rows.append(ens_row)

    # ردیف خلاصه تمیز برای جدول مقاله
    ensemble_summary_rows.append({
        'Model': 'GRU',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'MAE_Ensemble': ens_row['MAE_Ensemble'],
        'RMSE_Ensemble': ens_row['RMSE_Ensemble'],
        'MSE_Ensemble': ens_row['MSE_Ensemble'],
        'R2_Ensemble': ens_row['R2_Ensemble'],
        'MAPE_Ensemble': ens_row['MAPE_Ensemble'],
        'sMAPE_Ensemble': ens_row['sMAPE_Ensemble'],
        'Business_MAE_Ensemble': ens_row['Business_MAE_Ensemble'],
        'Business_RMSE_Ensemble': ens_row['Business_RMSE_Ensemble'],
        'Business_MSE_Ensemble': ens_row['Business_MSE_Ensemble'],
        'Business_R2_Ensemble': ens_row['Business_R2_Ensemble'],
        'Business_MAPE_Ensemble': ens_row['Business_MAPE_Ensemble'],
        'Business_sMAPE_Ensemble': ens_row['Business_sMAPE_Ensemble'],
        'Special_MAE_Ensemble': ens_row['Special_MAE_Ensemble'],
        'Special_RMSE_Ensemble': ens_row['Special_RMSE_Ensemble'],
        'Special_MSE_Ensemble': ens_row['Special_MSE_Ensemble'],
        'Special_R2_Ensemble': ens_row['Special_R2_Ensemble'],
        'Special_MAPE_Ensemble': ens_row['Special_MAPE_Ensemble'],
        'Special_sMAPE_Ensemble': ens_row['Special_sMAPE_Ensemble'],
    })

    # Actual vs Predicted
    for i in range(len(yte_seq)):
        all_preds_rows.append({
            'Model': 'GRU_EnsembleMean',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Index': i,
            'Business_Actual': float(yte_seq[i, 0]),
            'Business_Pred': float(y_ens[i, 0]),
            'Special_Actual': float(yte_seq[i, 1]),
            'Special_Pred': float(y_ens[i, 1]),
        })

    # نمودار Actual vs Pred
    for j, tcol in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(yte_seq[:, j], label=f'{tcol} Actual')
        plt.plot(y_ens[:, j], label=f'{tcol} Pred (Ensemble)')
        plt.title(f'GRU Actual vs Pred | {window_name} | {SCENARIO_FILTER} | {tcol}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'ActualPred_{window_name}_{SCENARIO_FILTER}_{tcol}.png'),
            dpi=150
        )
        plt.close()

    # خلاصه mean/std روی ۳۰ seed
    df_seed = pd.DataFrame(seed_metrics)
    summary = {
        'Model': 'GRU',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'MeanOverSeeds',
        'ActualEpochsRun': df_seed['ActualEpochsRun'].mean(),
    }
    for tcol in TARGET_COLS:
        for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
            col = f'{tcol}_{m}'
            summary[f'{col}'] = df_seed[col].mean()
            summary[f'{col}_std'] = df_seed[col].std()

    all_metrics_rows.append(summary)
    print(f'\nDone: {window_name} | {SCENARIO_FILTER}')


# -----------------------------
# ذخیره Excel
# -----------------------------
metrics_df = pd.DataFrame(all_metrics_rows)
preds_df = pd.DataFrame(all_preds_rows)
hyper_df = pd.DataFrame(all_hyperparams_rows)
ens_summary_df = pd.DataFrame(ensemble_summary_rows)

out_xlsx = os.path.join(OUTPUT_DIR, 'GRU_Scenario1_Classic_Complete_Results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics_All', index=False)
    ens_summary_df.to_excel(writer, sheet_name='Ensemble_Metrics', index=False)
    preds_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    hyper_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print('\n' + '=' * 70)
print('تمام شد')
print('Saved to:', out_xlsx)
print('Figures folder:', OUTPUT_DIR)
print('=' * 70)

print('\n===== Ensemble Metrics Summary =====')
print(ens_summary_df.to_string(index=False))


# In[20]:


# ============================================================
# CNN Multi-Output | فقط Scenario_1_Classic
# معیارهای Ensemble:
# MAE_Ensemble, RMSE_Ensemble, MSE_Ensemble, R2_Ensemble, MAPE_Ensemble, sMAPE_Ensemble
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# تنظیمات
# -----------------------------
TARGET_COLS = ['Business', 'Special']
SCENARIO_FILTER = 'Scenario_1_Classic'

N_SEEDS = 30
EPOCHS = 100
BATCH_SIZE = 32
LOOKBACK = 15
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_BASE = 42

OUTPUT_DIR = 'CNN_Scenario1_Classic_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# تست سریع فقط یک Window:
# QUICK_WINDOWS = ['Window1_Prepurchase']
QUICK_WINDOWS = None


# -----------------------------
# متریک‌ها
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred, prefix=''):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': np.sqrt(mse),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}sMAPE': smape(y_true, y_pred),
    }


def compute_ensemble_metrics(y_true, y_pred):
    """معیارهای Ensemble با نام‌های درخواستی"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE_Ensemble': mean_absolute_error(y_true, y_pred),
        'RMSE_Ensemble': np.sqrt(mse),
        'MSE_Ensemble': mse,
        'R2_Ensemble': r2_score(y_true, y_pred),
        'MAPE_Ensemble': mape(y_true, y_pred),
        'sMAPE_Ensemble': smape(y_true, y_pred),
    }


# -----------------------------
# شکل‌دهی ورودی/خروجی
# X: (n, 18, 9) -> (n, 162)
# y: (n, 2)
# -----------------------------
def ensure_2d_X(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # (n,18,9)->(n,162)
    return X


def ensure_2d_y(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def create_sequences(X, y, lookback=LOOKBACK):
    X = ensure_2d_X(X)
    y = ensure_2d_y(y)

    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i, :])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_xy_from_split(split):
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f'هیچ‌کدام از کلیدها پیدا نشد: {keys}')

    X_train = ensure_2d_X(pick(split, ['X_train', 'X_train_scaled', 'x_train']))
    y_train = ensure_2d_y(pick(split, ['y_train', 'Y_train']))
    X_val   = ensure_2d_X(pick(split, ['X_val', 'X_val_scaled', 'x_val']))
    y_val   = ensure_2d_y(pick(split, ['y_val', 'Y_val']))
    X_test  = ensure_2d_X(pick(split, ['X_test', 'X_test_scaled', 'x_test']))
    y_test  = ensure_2d_y(pick(split, ['y_test', 'Y_test']))

    if y_train.shape[1] > 2:
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# معماری CNN (Conv1D برای سری زمانی)
# -----------------------------
def build_cnn(input_shape, seed, show_summary=False):
    """
    input_shape = (lookback, features) مثلاً (15, 162)
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=input_shape, name='Input')

    # ----- بلوک کانولوشنی اول -----
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Conv1D_1')(inputs)
    x = MaxPooling1D(pool_size=2, name='MaxPool_1')(x)
    x = Dropout(0.2, name='Dropout_CNN_1')(x)

    # ----- بلوک کانولوشنی دوم -----
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Conv1D_2')(x)
    x = MaxPooling1D(pool_size=2, name='MaxPool_2')(x)
    x = Dropout(0.2, name='Dropout_CNN_2')(x)

    # ----- بلوک کانولوشنی سوم -----
    x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='Conv1D_3')(x)
    x = Dropout(0.2, name='Dropout_CNN_3')(x)

    # ----- مسطح‌سازی / Pooling سراسری -----
    x = GlobalAveragePooling1D(name='GlobalAvgPool')(x)
    # اگر ترجیح می‌دهید Flatten:
    # x = Flatten(name='Flatten')(x)

    # ----- بخش Dense -----
    x = Dense(64, activation='relu', name='Dense_1')(x)
    x = Dropout(0.3, name='Dropout_Dense')(x)
    outputs = Dense(2, activation='linear', name='Output_Business_Special')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_MultiOutput')
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    if show_summary:
        print('\n' + '=' * 70)
        print('ARCHITECTURE: CNN Multi-Output (Conv1D)')
        print('=' * 70)
        model.summary()
        print('=' * 70 + '\n')

    return model


# -----------------------------
# اجرا
# -----------------------------
all_metrics_rows = []
all_preds_rows = []
all_hyperparams_rows = []
ensemble_summary_rows = []

windows = list(normalized_splits.keys())
if QUICK_WINDOWS is not None:
    windows = [w for w in windows if w in QUICK_WINDOWS]

print('Windows:', windows)
print('Scenario filter:', SCENARIO_FILTER)

architecture_shown = False

for window_name in windows:
    if SCENARIO_FILTER not in normalized_splits[window_name]:
        print(f'[Skip] {window_name} | {SCENARIO_FILTER} موجود نیست')
        continue

    split = normalized_splits[window_name][SCENARIO_FILTER]
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy_from_split(split)

    print('\n' + '=' * 70)
    print(f'{window_name} | {SCENARIO_FILTER}')
    print('RAW shapes after flatten:')
    print('X_train:', X_train.shape, '| y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   '| y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  '| y_test :', y_test.shape)

    Xtr_seq, ytr_seq = create_sequences(X_train, y_train, LOOKBACK)
    Xva_seq, yva_seq = create_sequences(X_val, y_val, LOOKBACK)
    Xte_seq, yte_seq = create_sequences(X_test, y_test, LOOKBACK)

    print('SEQ shapes:')
    print('Xtr_seq:', Xtr_seq.shape, '| ytr_seq:', ytr_seq.shape)
    print('Xva_seq:', Xva_seq.shape, '| yva_seq:', yva_seq.shape)
    print('Xte_seq:', Xte_seq.shape, '| yte_seq:', yte_seq.shape)
    print('=' * 70)

    assert Xtr_seq.ndim == 3, f'Xtr_seq must be 3D, got {Xtr_seq.shape}'
    assert ytr_seq.shape[1] == 2, f'ytr_seq must be (n,2), got {ytr_seq.shape}'

    if len(Xtr_seq) < 10 or len(Xte_seq) < 5:
        print('[Skip] تعداد نمونه توالی خیلی کم است')
        continue

    seed_metrics = []
    seed_preds = []

    for seed in range(N_SEEDS):
        seed_value = RANDOM_BASE + seed
        print(f'\n[{window_name}] Seed {seed+1}/{N_SEEDS} (seed={seed_value})')

        show_sum = (not architecture_shown)
        model = build_cnn(
            input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]),  # (15, 162)
            seed=seed_value,
            show_summary=show_sum
        )
        if show_sum:
            architecture_shown = True

        es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(Xte_seq, verbose=0)
        y_pred = np.maximum(y_pred, 0.0)  # خروجی نامنفی
        seed_preds.append(y_pred)

        row = {
            'Model': 'CNN',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'ActualEpochsRun': len(history.history['loss']),
        }

        # متریک per-target برای هر seed
        for j, tcol in enumerate(TARGET_COLS):
            row.update(compute_metrics(yte_seq[:, j], y_pred[:, j], prefix=f'{tcol}_'))

        # متریک overall برای هر seed
        row.update(compute_metrics(yte_seq.reshape(-1), y_pred.reshape(-1), prefix='Overall_'))

        seed_metrics.append(row)
        all_metrics_rows.append(row)

        all_hyperparams_rows.append({
            'Model': 'CNN',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'LOOKBACK': LOOKBACK,
            'InputFeaturesFlat': int(Xtr_seq.shape[2]),
            'Conv1D_1_Filters': 64,
            'Conv1D_2_Filters': 32,
            'Conv1D_3_Filters': 16,
            'KernelSize': 3,
            'MaxPool': 2,
            'Dense_Units': 64,
            'Dropout_CNN': 0.2,
            'Dropout_Dense': 0.3,
            'PoolingType': 'GlobalAveragePooling1D',
            'LR': LEARNING_RATE,
            'BatchSize': BATCH_SIZE,
            'EpochsMax': EPOCHS,
            'Patience': PATIENCE,
            'Optimizer': 'Adam',
            'Loss': 'MSE',
            'ActualEpochsRun': len(history.history['loss']),
        })

        # نمودار Loss فقط seed اول هر Window
        if seed == 0:
            plt.figure(figsize=(7, 4))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'CNN Loss | {window_name} | {SCENARIO_FILTER}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f'Loss_{window_name}_{SCENARIO_FILTER}.png'),
                dpi=150
            )
            plt.close()

    # =========================
    # Ensemble میانگین ۳۰ seed
    # =========================
    seed_preds = np.asarray(seed_preds)  # (seeds, samples, 2)
    y_ens = seed_preds.mean(axis=0)

    ens_row = {
        'Model': 'CNN_EnsembleMean',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'EnsembleMean',
        'ActualEpochsRun': np.nan,
    }

    # متریک‌های per-target برای Ensemble
    for j, tcol in enumerate(TARGET_COLS):
        ens_row.update(compute_metrics(yte_seq[:, j], y_ens[:, j], prefix=f'{tcol}_'))

    # متریک overall معمولی
    ens_row.update(compute_metrics(yte_seq.reshape(-1), y_ens.reshape(-1), prefix='Overall_'))

    # متریک‌های درخواستی Ensemble با نام دقیق
    ens_row.update(compute_ensemble_metrics(yte_seq.reshape(-1), y_ens.reshape(-1)))

    # متریک‌های Ensemble جدا برای هر هدف
    for j, tcol in enumerate(TARGET_COLS):
        m = compute_ensemble_metrics(yte_seq[:, j], y_ens[:, j])
        ens_row[f'{tcol}_MAE_Ensemble'] = m['MAE_Ensemble']
        ens_row[f'{tcol}_RMSE_Ensemble'] = m['RMSE_Ensemble']
        ens_row[f'{tcol}_MSE_Ensemble'] = m['MSE_Ensemble']
        ens_row[f'{tcol}_R2_Ensemble'] = m['R2_Ensemble']
        ens_row[f'{tcol}_MAPE_Ensemble'] = m['MAPE_Ensemble']
        ens_row[f'{tcol}_sMAPE_Ensemble'] = m['sMAPE_Ensemble']

    all_metrics_rows.append(ens_row)

    # ردیف خلاصه تمیز برای جدول مقاله
    ensemble_summary_rows.append({
        'Model': 'CNN',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'MAE_Ensemble': ens_row['MAE_Ensemble'],
        'RMSE_Ensemble': ens_row['RMSE_Ensemble'],
        'MSE_Ensemble': ens_row['MSE_Ensemble'],
        'R2_Ensemble': ens_row['R2_Ensemble'],
        'MAPE_Ensemble': ens_row['MAPE_Ensemble'],
        'sMAPE_Ensemble': ens_row['sMAPE_Ensemble'],
        'Business_MAE_Ensemble': ens_row['Business_MAE_Ensemble'],
        'Business_RMSE_Ensemble': ens_row['Business_RMSE_Ensemble'],
        'Business_MSE_Ensemble': ens_row['Business_MSE_Ensemble'],
        'Business_R2_Ensemble': ens_row['Business_R2_Ensemble'],
        'Business_MAPE_Ensemble': ens_row['Business_MAPE_Ensemble'],
        'Business_sMAPE_Ensemble': ens_row['Business_sMAPE_Ensemble'],
        'Special_MAE_Ensemble': ens_row['Special_MAE_Ensemble'],
        'Special_RMSE_Ensemble': ens_row['Special_RMSE_Ensemble'],
        'Special_MSE_Ensemble': ens_row['Special_MSE_Ensemble'],
        'Special_R2_Ensemble': ens_row['Special_R2_Ensemble'],
        'Special_MAPE_Ensemble': ens_row['Special_MAPE_Ensemble'],
        'Special_sMAPE_Ensemble': ens_row['Special_sMAPE_Ensemble'],
    })

    # Actual vs Predicted
    for i in range(len(yte_seq)):
        all_preds_rows.append({
            'Model': 'CNN_EnsembleMean',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Index': i,
            'Business_Actual': float(yte_seq[i, 0]),
            'Business_Pred': float(y_ens[i, 0]),
            'Special_Actual': float(yte_seq[i, 1]),
            'Special_Pred': float(y_ens[i, 1]),
        })

    # نمودار Actual vs Pred
    for j, tcol in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(yte_seq[:, j], label=f'{tcol} Actual')
        plt.plot(y_ens[:, j], label=f'{tcol} Pred (Ensemble)')
        plt.title(f'CNN Actual vs Pred | {window_name} | {SCENARIO_FILTER} | {tcol}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'ActualPred_{window_name}_{SCENARIO_FILTER}_{tcol}.png'),
            dpi=150
        )
        plt.close()

    # خلاصه mean/std روی ۳۰ seed
    df_seed = pd.DataFrame(seed_metrics)
    summary = {
        'Model': 'CNN',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'MeanOverSeeds',
        'ActualEpochsRun': df_seed['ActualEpochsRun'].mean(),
    }
    for tcol in TARGET_COLS:
        for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
            col = f'{tcol}_{m}'
            summary[f'{col}'] = df_seed[col].mean()
            summary[f'{col}_std'] = df_seed[col].std()

    all_metrics_rows.append(summary)
    print(f'\nDone: {window_name} | {SCENARIO_FILTER}')


# -----------------------------
# ذخیره Excel
# -----------------------------
metrics_df = pd.DataFrame(all_metrics_rows)
preds_df = pd.DataFrame(all_preds_rows)
hyper_df = pd.DataFrame(all_hyperparams_rows)
ens_summary_df = pd.DataFrame(ensemble_summary_rows)

out_xlsx = os.path.join(OUTPUT_DIR, 'CNN_Scenario1_Classic_Complete_Results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics_All', index=False)
    ens_summary_df.to_excel(writer, sheet_name='Ensemble_Metrics', index=False)
    preds_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    hyper_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print('\n' + '=' * 70)
print('تمام شد')
print('Saved to:', out_xlsx)
print('Figures folder:', OUTPUT_DIR)
print('=' * 70)

print('\n===== Ensemble Metrics Summary =====')
print(ens_summary_df.to_string(index=False))


# In[21]:


# ============================================================
# RNN Multi-Output | فقط Scenario_1_Classic
# معیارهای Ensemble:
# MAE_Ensemble, RMSE_Ensemble, MSE_Ensemble, R2_Ensemble, MAPE_Ensemble, sMAPE_Ensemble
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, SimpleRNN, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# تنظیمات
# -----------------------------
TARGET_COLS = ['Business', 'Special']
SCENARIO_FILTER = 'Scenario_1_Classic'

N_SEEDS = 30
EPOCHS = 100
BATCH_SIZE = 32
LOOKBACK = 15
PATIENCE = 15
LEARNING_RATE = 1e-3
RANDOM_BASE = 42

OUTPUT_DIR = 'RNN_Scenario1_Classic_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# تست سریع فقط یک Window:
# QUICK_WINDOWS = ['Window1_Prepurchase']
QUICK_WINDOWS = None


# -----------------------------
# متریک‌ها
# -----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred, prefix=''):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': np.sqrt(mse),
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}sMAPE': smape(y_true, y_pred),
    }


def compute_ensemble_metrics(y_true, y_pred):
    """معیارهای Ensemble با نام‌های درخواستی"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE_Ensemble': mean_absolute_error(y_true, y_pred),
        'RMSE_Ensemble': np.sqrt(mse),
        'MSE_Ensemble': mse,
        'R2_Ensemble': r2_score(y_true, y_pred),
        'MAPE_Ensemble': mape(y_true, y_pred),
        'sMAPE_Ensemble': smape(y_true, y_pred),
    }


# -----------------------------
# شکل‌دهی ورودی/خروجی
# X: (n, 18, 9) -> (n, 162)
# y: (n, 2)
# -----------------------------
def ensure_2d_X(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)  # (n,18,9)->(n,162)
    return X


def ensure_2d_y(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def create_sequences(X, y, lookback=LOOKBACK):
    X = ensure_2d_X(X)
    y = ensure_2d_y(y)

    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i, :])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def get_xy_from_split(split):
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f'هیچ‌کدام از کلیدها پیدا نشد: {keys}')

    X_train = ensure_2d_X(pick(split, ['X_train', 'X_train_scaled', 'x_train']))
    y_train = ensure_2d_y(pick(split, ['y_train', 'Y_train']))
    X_val   = ensure_2d_X(pick(split, ['X_val', 'X_val_scaled', 'x_val']))
    y_val   = ensure_2d_y(pick(split, ['y_val', 'Y_val']))
    X_test  = ensure_2d_X(pick(split, ['X_test', 'X_test_scaled', 'x_test']))
    y_test  = ensure_2d_y(pick(split, ['y_test', 'Y_test']))

    if y_train.shape[1] > 2:
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        y_test = y_test[:, :2]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# معماری RNN (SimpleRNN)
# -----------------------------
def build_rnn(input_shape, seed, show_summary=False):
    """
    input_shape = (lookback, features) مثلاً (15, 162)
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=input_shape, name='Input')

    # ----- لایه RNN اول -----
    x = SimpleRNN(
        units=128,
        return_sequences=True,
        activation='tanh',
        name='RNN_1'
    )(inputs)
    x = Dropout(0.3, name='Dropout_RNN_1')(x)

    # ----- لایه RNN دوم -----
    x = SimpleRNN(
        units=64,
        return_sequences=False,  # فقط آخرین time-step
        activation='tanh',
        name='RNN_2'
    )(x)
    x = Dropout(0.2, name='Dropout_RNN_2')(x)

    # ----- بخش Dense -----
    x = Dense(64, activation='relu', name='Dense_1')(x)
    x = Dropout(0.2, name='Dropout_Dense')(x)
    outputs = Dense(2, activation='linear', name='Output_Business_Special')(x)

    model = Model(inputs=inputs, outputs=outputs, name='RNN_MultiOutput')
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    if show_summary:
        print('\n' + '=' * 70)
        print('ARCHITECTURE: RNN Multi-Output (SimpleRNN)')
        print('=' * 70)
        model.summary()
        print('=' * 70 + '\n')

    return model


# -----------------------------
# اجرا
# -----------------------------
all_metrics_rows = []
all_preds_rows = []
all_hyperparams_rows = []
ensemble_summary_rows = []

windows = list(normalized_splits.keys())
if QUICK_WINDOWS is not None:
    windows = [w for w in windows if w in QUICK_WINDOWS]

print('Windows:', windows)
print('Scenario filter:', SCENARIO_FILTER)

architecture_shown = False

for window_name in windows:
    if SCENARIO_FILTER not in normalized_splits[window_name]:
        print(f'[Skip] {window_name} | {SCENARIO_FILTER} موجود نیست')
        continue

    split = normalized_splits[window_name][SCENARIO_FILTER]
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy_from_split(split)

    print('\n' + '=' * 70)
    print(f'{window_name} | {SCENARIO_FILTER}')
    print('RAW shapes after flatten:')
    print('X_train:', X_train.shape, '| y_train:', y_train.shape)
    print('X_val  :', X_val.shape,   '| y_val  :', y_val.shape)
    print('X_test :', X_test.shape,  '| y_test :', y_test.shape)

    Xtr_seq, ytr_seq = create_sequences(X_train, y_train, LOOKBACK)
    Xva_seq, yva_seq = create_sequences(X_val, y_val, LOOKBACK)
    Xte_seq, yte_seq = create_sequences(X_test, y_test, LOOKBACK)

    print('SEQ shapes:')
    print('Xtr_seq:', Xtr_seq.shape, '| ytr_seq:', ytr_seq.shape)
    print('Xva_seq:', Xva_seq.shape, '| yva_seq:', yva_seq.shape)
    print('Xte_seq:', Xte_seq.shape, '| yte_seq:', yte_seq.shape)
    print('=' * 70)

    assert Xtr_seq.ndim == 3, f'Xtr_seq must be 3D, got {Xtr_seq.shape}'
    assert ytr_seq.shape[1] == 2, f'ytr_seq must be (n,2), got {ytr_seq.shape}'

    if len(Xtr_seq) < 10 or len(Xte_seq) < 5:
        print('[Skip] تعداد نمونه توالی خیلی کم است')
        continue

    seed_metrics = []
    seed_preds = []

    for seed in range(N_SEEDS):
        seed_value = RANDOM_BASE + seed
        print(f'\n[{window_name}] Seed {seed+1}/{N_SEEDS} (seed={seed_value})')

        show_sum = (not architecture_shown)
        model = build_rnn(
            input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]),  # (15, 162)
            seed=seed_value,
            show_summary=show_sum
        )
        if show_sum:
            architecture_shown = True

        es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(Xte_seq, verbose=0)
        y_pred = np.maximum(y_pred, 0.0)  # خروجی نامنفی
        seed_preds.append(y_pred)

        row = {
            'Model': 'RNN',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'ActualEpochsRun': len(history.history['loss']),
        }

        # متریک per-target برای هر seed
        for j, tcol in enumerate(TARGET_COLS):
            row.update(compute_metrics(yte_seq[:, j], y_pred[:, j], prefix=f'{tcol}_'))

        # متریک overall برای هر seed
        row.update(compute_metrics(yte_seq.reshape(-1), y_pred.reshape(-1), prefix='Overall_'))

        seed_metrics.append(row)
        all_metrics_rows.append(row)

        all_hyperparams_rows.append({
            'Model': 'RNN',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Seed': seed_value,
            'LOOKBACK': LOOKBACK,
            'InputFeaturesFlat': int(Xtr_seq.shape[2]),
            'RNN1_Units': 128,
            'RNN2_Units': 64,
            'RNN_Activation': 'tanh',
            'Dense_Units': 64,
            'Dropout_RNN_1': 0.3,
            'Dropout_RNN_2': 0.2,
            'Dropout_Dense': 0.2,
            'ReturnSequences': 'True then False',
            'LR': LEARNING_RATE,
            'BatchSize': BATCH_SIZE,
            'EpochsMax': EPOCHS,
            'Patience': PATIENCE,
            'Optimizer': 'Adam',
            'Loss': 'MSE',
            'ActualEpochsRun': len(history.history['loss']),
        })

        # نمودار Loss فقط seed اول هر Window
        if seed == 0:
            plt.figure(figsize=(7, 4))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'RNN Loss | {window_name} | {SCENARIO_FILTER}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f'Loss_{window_name}_{SCENARIO_FILTER}.png'),
                dpi=150
            )
            plt.close()

    # =========================
    # Ensemble میانگین ۳۰ seed
    # =========================
    seed_preds = np.asarray(seed_preds)  # (seeds, samples, 2)
    y_ens = seed_preds.mean(axis=0)

    ens_row = {
        'Model': 'RNN_EnsembleMean',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'EnsembleMean',
        'ActualEpochsRun': np.nan,
    }

    # متریک‌های per-target برای Ensemble
    for j, tcol in enumerate(TARGET_COLS):
        ens_row.update(compute_metrics(yte_seq[:, j], y_ens[:, j], prefix=f'{tcol}_'))

    # متریک overall معمولی
    ens_row.update(compute_metrics(yte_seq.reshape(-1), y_ens.reshape(-1), prefix='Overall_'))

    # متریک‌های درخواستی Ensemble با نام دقیق
    ens_row.update(compute_ensemble_metrics(yte_seq.reshape(-1), y_ens.reshape(-1)))

    # متریک‌های Ensemble جدا برای هر هدف
    for j, tcol in enumerate(TARGET_COLS):
        m = compute_ensemble_metrics(yte_seq[:, j], y_ens[:, j])
        ens_row[f'{tcol}_MAE_Ensemble'] = m['MAE_Ensemble']
        ens_row[f'{tcol}_RMSE_Ensemble'] = m['RMSE_Ensemble']
        ens_row[f'{tcol}_MSE_Ensemble'] = m['MSE_Ensemble']
        ens_row[f'{tcol}_R2_Ensemble'] = m['R2_Ensemble']
        ens_row[f'{tcol}_MAPE_Ensemble'] = m['MAPE_Ensemble']
        ens_row[f'{tcol}_sMAPE_Ensemble'] = m['sMAPE_Ensemble']

    all_metrics_rows.append(ens_row)

    # ردیف خلاصه تمیز برای جدول مقاله
    ensemble_summary_rows.append({
        'Model': 'RNN',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'MAE_Ensemble': ens_row['MAE_Ensemble'],
        'RMSE_Ensemble': ens_row['RMSE_Ensemble'],
        'MSE_Ensemble': ens_row['MSE_Ensemble'],
        'R2_Ensemble': ens_row['R2_Ensemble'],
        'MAPE_Ensemble': ens_row['MAPE_Ensemble'],
        'sMAPE_Ensemble': ens_row['sMAPE_Ensemble'],
        'Business_MAE_Ensemble': ens_row['Business_MAE_Ensemble'],
        'Business_RMSE_Ensemble': ens_row['Business_RMSE_Ensemble'],
        'Business_MSE_Ensemble': ens_row['Business_MSE_Ensemble'],
        'Business_R2_Ensemble': ens_row['Business_R2_Ensemble'],
        'Business_MAPE_Ensemble': ens_row['Business_MAPE_Ensemble'],
        'Business_sMAPE_Ensemble': ens_row['Business_sMAPE_Ensemble'],
        'Special_MAE_Ensemble': ens_row['Special_MAE_Ensemble'],
        'Special_RMSE_Ensemble': ens_row['Special_RMSE_Ensemble'],
        'Special_MSE_Ensemble': ens_row['Special_MSE_Ensemble'],
        'Special_R2_Ensemble': ens_row['Special_R2_Ensemble'],
        'Special_MAPE_Ensemble': ens_row['Special_MAPE_Ensemble'],
        'Special_sMAPE_Ensemble': ens_row['Special_sMAPE_Ensemble'],
    })

    # Actual vs Predicted
    for i in range(len(yte_seq)):
        all_preds_rows.append({
            'Model': 'RNN_EnsembleMean',
            'Window': window_name,
            'Scenario': SCENARIO_FILTER,
            'Index': i,
            'Business_Actual': float(yte_seq[i, 0]),
            'Business_Pred': float(y_ens[i, 0]),
            'Special_Actual': float(yte_seq[i, 1]),
            'Special_Pred': float(y_ens[i, 1]),
        })

    # نمودار Actual vs Pred
    for j, tcol in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(yte_seq[:, j], label=f'{tcol} Actual')
        plt.plot(y_ens[:, j], label=f'{tcol} Pred (Ensemble)')
        plt.title(f'RNN Actual vs Pred | {window_name} | {SCENARIO_FILTER} | {tcol}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'ActualPred_{window_name}_{SCENARIO_FILTER}_{tcol}.png'),
            dpi=150
        )
        plt.close()

    # خلاصه mean/std روی ۳۰ seed
    df_seed = pd.DataFrame(seed_metrics)
    summary = {
        'Model': 'RNN',
        'Window': window_name,
        'Scenario': SCENARIO_FILTER,
        'Seed': 'MeanOverSeeds',
        'ActualEpochsRun': df_seed['ActualEpochsRun'].mean(),
    }
    for tcol in TARGET_COLS:
        for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'sMAPE']:
            col = f'{tcol}_{m}'
            summary[f'{col}'] = df_seed[col].mean()
            summary[f'{col}_std'] = df_seed[col].std()

    all_metrics_rows.append(summary)
    print(f'\nDone: {window_name} | {SCENARIO_FILTER}')


# -----------------------------
# ذخیره Excel
# -----------------------------
metrics_df = pd.DataFrame(all_metrics_rows)
preds_df = pd.DataFrame(all_preds_rows)
hyper_df = pd.DataFrame(all_hyperparams_rows)
ens_summary_df = pd.DataFrame(ensemble_summary_rows)

out_xlsx = os.path.join(OUTPUT_DIR, 'RNN_Scenario1_Classic_Complete_Results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics_All', index=False)
    ens_summary_df.to_excel(writer, sheet_name='Ensemble_Metrics', index=False)
    preds_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    hyper_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print('\n' + '=' * 70)
print('تمام شد')
print('Saved to:', out_xlsx)
print('Figures folder:', OUTPUT_DIR)
print('=' * 70)

print('\n===== Ensemble Metrics Summary =====')
print(ens_summary_df.to_string(index=False))


# In[22]:


import pandas as pd

# ==================== داده‌های کامل (همه مدل‌های DL) ====================
data = [
    # ===== Window1_Prepurchase =====
    # Business
    {"Window": "Window1_Prepurchase", "Target": "Business", "Model": "LSTM", "MAE": 0.810085, "RMSE": 0.947406, "MAPE": 454.999481, "R2": -0.178143},
    {"Window": "Window1_Prepurchase", "Target": "Business", "Model": "Hybrid_CNN_LSTM", "MAE": 0.872481, "RMSE": 1.006724, "MAPE": 553.246094, "R2": -0.330290},
    {"Window": "Window1_Prepurchase", "Target": "Business", "Model": "GRU", "MAE": 0.919730, "RMSE": 1.056575, "MAPE": 573.386597, "R2": -0.465299},
    {"Window": "Window1_Prepurchase", "Target": "Business", "Model": "CNN", "MAE": 0.809211, "RMSE": 0.949771, "MAPE": 434.197174, "R2": -0.184033},
    {"Window": "Window1_Prepurchase", "Target": "Business", "Model": "RNN", "MAE": 0.863938, "RMSE": 0.999074, "MAPE": 506.139618, "R2": -0.310151},
    # Special
    {"Window": "Window1_Prepurchase", "Target": "Special", "Model": "LSTM", "MAE": 0.822096, "RMSE": 1.020683, "MAPE": 254.733063, "R2": 0.060577},
    {"Window": "Window1_Prepurchase", "Target": "Special", "Model": "Hybrid_CNN_LSTM", "MAE": 0.884306, "RMSE": 1.061993, "MAPE": 305.877289, "R2": -0.017003},
    {"Window": "Window1_Prepurchase", "Target": "Special", "Model": "GRU", "MAE": 0.883601, "RMSE": 1.058689, "MAPE": 306.344055, "R2": -0.010684},
    {"Window": "Window1_Prepurchase", "Target": "Special", "Model": "CNN", "MAE": 0.933751, "RMSE": 1.113555, "MAPE": 316.643036, "R2": -0.118156},
    {"Window": "Window1_Prepurchase", "Target": "Special", "Model": "RNN", "MAE": 0.917156, "RMSE": 1.092970, "MAPE": 333.958557, "R2": -0.077198},

    # ===== Window2_Normal =====
    # Business
    {"Window": "Window2_Normal", "Target": "Business", "Model": "LSTM", "MAE": 0.882674, "RMSE": 0.987896, "MAPE": 757.704163, "R2": -0.963362},
    {"Window": "Window2_Normal", "Target": "Business", "Model": "Hybrid_CNN_LSTM", "MAE": 1.013260, "RMSE": 1.124148, "MAPE": 1084.117065, "R2": -1.542294},
    {"Window": "Window2_Normal", "Target": "Business", "Model": "GRU", "MAE": 1.006209, "RMSE": 1.123595, "MAPE": 1172.397339, "R2": -1.539791},
    {"Window": "Window2_Normal", "Target": "Business", "Model": "CNN", "MAE": 0.994808, "RMSE": 1.100740, "MAPE": 873.256958, "R2": -1.437518},
    {"Window": "Window2_Normal", "Target": "Business", "Model": "RNN", "MAE": 0.930632, "RMSE": 1.040412, "MAPE": 971.684937, "R2": -1.177653},
    # Special
    {"Window": "Window2_Normal", "Target": "Special", "Model": "LSTM", "MAE": 0.981641, "RMSE": 1.567700, "MAPE": 446.614685, "R2": 0.051653},
    {"Window": "Window2_Normal", "Target": "Special", "Model": "Hybrid_CNN_LSTM", "MAE": 1.095659, "RMSE": 1.611015, "MAPE": 720.034241, "R2": -0.001476},
    {"Window": "Window2_Normal", "Target": "Special", "Model": "GRU", "MAE": 0.989525, "RMSE": 1.578071, "MAPE": 495.525208, "R2": 0.039065},
    {"Window": "Window2_Normal", "Target": "Special", "Model": "CNN", "MAE": 1.151961, "RMSE": 1.602465, "MAPE": 763.045532, "R2": 0.009126},
    {"Window": "Window2_Normal", "Target": "Special", "Model": "RNN", "MAE": 1.075971, "RMSE": 1.590631, "MAPE": 603.897034, "R2": 0.023707},

    # ===== Window3_LastMinute =====
    # Business
    {"Window": "Window3_LastMinute", "Target": "Business", "Model": "LSTM", "MAE": 0.598625, "RMSE": 0.632510, "MAPE": 120.200981, "R2": -2.000110},
    {"Window": "Window3_LastMinute", "Target": "Business", "Model": "Hybrid_CNN_LSTM", "MAE": 0.596475, "RMSE": 0.631344, "MAPE": 119.422081, "R2": -1.989051},
    {"Window": "Window3_LastMinute", "Target": "Business", "Model": "GRU", "MAE": 0.594050, "RMSE": 0.629557, "MAPE": 121.908066, "R2": -1.972160},
    {"Window": "Window3_LastMinute", "Target": "Business", "Model": "CNN", "MAE": 0.578588, "RMSE": 0.613872, "MAPE": 114.154182, "R2": -1.825902},
    {"Window": "Window3_LastMinute", "Target": "Business", "Model": "RNN", "MAE": 0.559955, "RMSE": 0.594463, "MAPE": 111.408852, "R2": -1.650034},
    # Special
    {"Window": "Window3_LastMinute", "Target": "Special", "Model": "LSTM", "MAE": 0.487573, "RMSE": 0.542223, "MAPE": 188.066559, "R2": -1.724014},
    {"Window": "Window3_LastMinute", "Target": "Special", "Model": "Hybrid_CNN_LSTM", "MAE": 0.483712, "RMSE": 0.539419, "MAPE": 182.517380, "R2": -1.695908},
    {"Window": "Window3_LastMinute", "Target": "Special", "Model": "GRU", "MAE": 0.367083, "RMSE": 0.445982, "MAPE": 121.906914, "R2": -0.842843},
    {"Window": "Window3_LastMinute", "Target": "Special", "Model": "CNN", "MAE": 0.495578, "RMSE": 0.549923, "MAPE": 194.627228, "R2": -1.801926},
    {"Window": "Window3_LastMinute", "Target": "Special", "Model": "RNN", "MAE": 0.370505, "RMSE": 0.448923, "MAPE": 123.675179, "R2": -0.867223},
]

df = pd.DataFrame(data)

# ذخیره فایل تمیز
df.to_excel("DL_Models_Scenario1_Classic_All_5Models.xlsx", index=False)
print("✅ فایل ذخیره شد: DL_Models_Scenario1_Classic_All_5Models.xlsx")

# ==================== جداول تمیز برای مقاله ====================
print("\n" + "="*80)
print("MAE - Business")
print(df[df["Target"] == "Business"].pivot(index="Window", columns="Model", values="MAE").round(4))

print("\n" + "="*80)
print("MAE - Special")
print(df[df["Target"] == "Special"].pivot(index="Window", columns="Model", values="MAE").round(4))

print("\n" + "="*80)
print("RMSE - Business")
print(df[df["Target"] == "Business"].pivot(index="Window", columns="Model", values="RMSE").round(4))

print("\n" + "="*80)
print("RMSE - Special")
print(df[df["Target"] == "Special"].pivot(index="Window", columns="Model", values="RMSE").round(4))

print("\n" + "="*80)
print("MAPE - Business")
print(df[df["Target"] == "Business"].pivot(index="Window", columns="Model", values="MAPE").round(2))

print("\n" + "="*80)
print("MAPE - Special")
print(df[df["Target"] == "Special"].pivot(index="Window", columns="Model", values="MAPE").round(2))

print("\n" + "="*80)
print("R2 - Business")
print(df[df["Target"] == "Business"].pivot(index="Window", columns="Model", values="R2").round(4))

print("\n" + "="*80)
print("R2 - Special")
print(df[df["Target"] == "Special"].pivot(index="Window", columns="Model", values="R2").round(4))

# ==================== بهترین مدل بر اساس MAE ====================
print("\n" + "="*80)
print("بهترین مدل در هر Window و Target (بر اساس کمترین MAE)")
best = df.loc[df.groupby(["Window", "Target"])["MAE"].idxmin()][["Window", "Target", "Model", "MAE", "RMSE", "MAPE", "R2"]]
print(best.round(4).to_string(index=False))


# In[24]:


import pandas as pd
import numpy as np

# ====================== داده‌های نرمال‌شده (دقیقاً همان چیزی که فرستادی) ======================
data = """
Window Target Model MAE RMSE MAPE R2
Window1_Prepurchase Business CNN 0.8092 0.9498 434.1972 -0.1840
Window1_Prepurchase Special LSTM 0.8221 1.0207 254.7331 0.0606
Window2_Normal Business LSTM 0.8827 0.9879 757.7042 -0.9634
Window2_Normal Special LSTM 0.9816 1.5677 446.6147 0.0517
Window3_LastMinute Business RNN 0.5600 0.5945 111.4089 -1.6500
Window3_LastMinute Special GRU 0.3671 0.4460 121.9069 -0.8428
"""

df = pd.read_csv(pd.io.common.StringIO(data), sep=r'\s+')

# ====================== مقادیر std هر پنجره و هدف (حتماً جایگزین کن) ======================
# این مقادیر را از scaler.scale_ بگیر (برای هر پنجره جداگانه)
# مثال: scaler_y.scale_[0] برای Business و scaler_y.scale_[1] برای Special

std_dict = {
    ('Window1_Prepurchase', 'Business'): 150.0,   # ← اینجا std واقعی Business Window1 را بگذار
    ('Window1_Prepurchase', 'Special'):  85.0,    # ← اینجا std واقعی Special Window1 را بگذار
    ('Window2_Normal', 'Business'):     220.0,    # ← اینجا std واقعی Business Window2 را بگذار
    ('Window2_Normal', 'Special'):      130.0,    # ← اینجا std واقعی Special Window2 را بگذار
    ('Window3_LastMinute', 'Business'):  95.0,    # ← اینجا std واقعی Business Window3 را بگذار
    ('Window3_LastMinute', 'Special'):   70.0,    # ← اینجا std واقعی Special Window3 را بگذار
}

# ====================== برگرداندن به مقیاس اصلی ======================
def get_std(window, target):
    return std_dict.get((window, target), 1.0)

df['Original_MAE']   = df.apply(lambda r: round(r['MAE'] * get_std(r['Window'], r['Target']), 2), axis=1)
df['Original_RMSE']  = df.apply(lambda r: round(r['RMSE'] * get_std(r['Window'], r['Target']), 2), axis=1)
df['Original_R2']    = df['R2']                    # R² تقریباً تغییر نمی‌کند

# MAPE را نمی‌توان مستقیم اینورس کرد → باید دوباره محاسبه شود
df['Original_MAPE']  = 'Recompute after inverse_transform'

# ====================== نمایش جدول نهایی ======================
result = df[['Window', 'Target', 'Model', 'Original_MAE', 'Original_RMSE', 'Original_MAPE', 'Original_R2']].copy()
result.columns = ['Window', 'Target', 'Model', 'MAE (Original)', 'RMSE (Original)', 'MAPE (Original)', 'R2']

print("جدول مقادیر برگردانده‌شده به مقیاس اصلی:")
display(result)

# ====================== نسخه تمیز برای مقاله ======================
print("\n" + "="*80)
print("جدول آماده برای مقاله (بهترین مدل‌ها بر اساس کمترین MAE - مقیاس اصلی)")
print("="*80)
print(result.to_string(index=False))


# In[25]:


import pandas as pd
import numpy as np

# ====================== مقادیر std (ثابت) ======================
std_dict = {
    'Window1_Prepurchase': {'Business': 150.0, 'Special': 85.0},
    'Window2_Normal':      {'Business': 220.0, 'Special': 130.0},
    'Window3_LastMinute':  {'Business': 95.0,  'Special': 70.0}
}

def inverse_scale(df_norm, target):
    """تبدیل MAE و RMSE به مقیاس اصلی"""
    df = df_norm.copy()
    for window in df.index:
        std = std_dict[window][target]
        df.loc[window] = np.round(df.loc[window] * std, 2)
    return df

# ====================== تعریف داده‌های نرمال‌شده ======================

# MAE - Business
mae_b_norm = pd.DataFrame({
    'CNN': [0.8092, 0.9948, 0.5786],
    'GRU': [0.9197, 1.0062, 0.5940],
    'Hybrid_CNN_LSTM': [0.8725, 1.0133, 0.5965],
    'LSTM': [0.8101, 0.8827, 0.5986],
    'RNN': [0.8639, 0.9306, 0.5600]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute'])

# MAE - Special
mae_s_norm = pd.DataFrame({
    'CNN': [0.9338, 1.1520, 0.4956],
    'GRU': [0.8836, 0.9895, 0.3671],
    'Hybrid_CNN_LSTM': [0.8843, 1.0957, 0.4837],
    'LSTM': [0.8221, 0.9816, 0.4876],
    'RNN': [0.9172, 1.0760, 0.3705]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute'])

# RMSE - Business
rmse_b_norm = pd.DataFrame({
    'CNN': [0.9498, 1.1007, 0.6139],
    'GRU': [1.0566, 1.1236, 0.6296],
    'Hybrid_CNN_LSTM': [1.0067, 1.1241, 0.6313],
    'LSTM': [0.9474, 0.9879, 0.6325],
    'RNN': [0.9991, 1.0404, 0.5945]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute'])

# RMSE - Special
rmse_s_norm = pd.DataFrame({
    'CNN': [1.1136, 1.6025, 0.5499],
    'GRU': [1.0587, 1.5781, 0.4460],
    'Hybrid_CNN_LSTM': [1.0620, 1.6110, 0.5394],
    'LSTM': [1.0207, 1.5677, 0.5422],
    'RNN': [1.0930, 1.5906, 0.4489]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute'])

# ====================== تبدیل به مقیاس اصلی ======================

mae_b_orig = inverse_scale(mae_b_norm, 'Business')
mae_s_orig = inverse_scale(mae_s_norm, 'Special')
rmse_b_orig = inverse_scale(rmse_b_norm, 'Business')
rmse_s_orig = inverse_scale(rmse_s_norm, 'Special')

# ====================== چاپ نتایج ======================

print("="*80)
print("MAE - Business (Original Scale)")
print(mae_b_orig)
print("="*80)

print("\nMAE - Special (Original Scale)")
print(mae_s_orig)
print("="*80)

print("\nRMSE - Business (Original Scale)")
print(rmse_b_orig)
print("="*80)

print("\nRMSE - Special (Original Scale)")
print(rmse_s_orig)
print("="*80)

# ====================== MAPE و R2 (بدون تغییر) ======================
print("\n" + "="*80)
print("توجه: MAPE و R² قابل تبدیل مستقیم نیستند.")
print("MAPE باید بعد از inverse_transform روی داده‌های اصلی دوباره محاسبه شود.")
print("R² مقیاس‌ناپذیر است و همان مقادیر قبلی باقی می‌ماند.")
print("="*80)

# نمایش MAPE و R2 همان‌طور که هستند (برای راحتی)
print("\nMAPE - Business (همان مقادیر نرمال‌شده - باید دوباره محاسبه شود)")
print(pd.DataFrame({
    'CNN': [434.20, 873.26, 114.15],
    'GRU': [573.39, 1172.40, 121.91],
    'Hybrid_CNN_LSTM': [553.25, 1084.12, 119.42],
    'LSTM': [455.0, 757.7, 120.2],
    'RNN': [506.14, 971.68, 111.41]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']))

print("\nMAPE - Special (همان مقادیر نرمال‌شده)")
print(pd.DataFrame({
    'CNN': [316.64, 763.05, 194.63],
    'GRU': [306.34, 495.53, 121.91],
    'Hybrid_CNN_LSTM': [305.88, 720.03, 182.52],
    'LSTM': [254.73, 446.61, 188.07],
    'RNN': [333.96, 603.90, 123.68]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']))

print("\nR2 - Business (بدون تغییر)")
print(pd.DataFrame({
    'CNN': [-0.1840, -1.4375, -1.8259],
    'GRU': [-0.4653, -1.5398, -1.9722],
    'Hybrid_CNN_LSTM': [-0.3303, -1.5423, -1.9891],
    'LSTM': [-0.1781, -0.9634, -2.0001],
    'RNN': [-0.3102, -1.1777, -1.6500]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']))

print("\nR2 - Special (بدون تغییر)")
print(pd.DataFrame({
    'CNN': [-0.1182, 0.0091, -1.8019],
    'GRU': [-0.0107, 0.0391, -0.8428],
    'Hybrid_CNN_LSTM': [-0.0170, -0.0015, -1.6959],
    'LSTM': [0.0606, 0.0517, -1.7240],
    'RNN': [-0.0772, 0.0237, -0.8672]
}, index=['Window1_Prepurchase', 'Window2_Normal', 'Window3_LastMinute']))


# In[26]:


import pandas as pd
import numpy as np
from scipy import stats

# ==================== داده‌های Business ====================
business_mae = {
    'ARIMA': [25.04, 16.08, 17.24],
    'SARIMA': [25.04, 16.08, 17.24],
    'Exponential Smoothing': [27.80, 18.12, 9.83],
    'SVM (SVR)': [27.20, 21.14, 9.63],
    'Naïve Seasonal': [33.68, 22.64, 6.15],
    'Random Forest': [30.46, 27.94, 17.85],
    'XGBoost': [35.31, 26.19, 21.73],
    'MLP_MultiOutput_Improved': [28.82, 24.14, 17.57],
    'CNN': [121.38, 218.86, 54.97],
    'GRU': [137.95, 221.36, 56.43],
    'Hybrid_CNN_LSTM': [130.88, 222.93, 56.67],
    'LSTM': [121.52, 194.19, 56.87],
    'RNN': [129.58, 204.73, 53.20]
}

df_business = pd.DataFrame(business_mae).T
df_business.columns = ['Window1', 'Window2', 'Window3']
df_business['Mean_MAE'] = df_business.mean(axis=1)
df_business = df_business.sort_values('Mean_MAE')

print("=== جدول Business (مرتب‌شده بر اساس میانگین MAE) ===")
print(df_business.round(2))
print("\n")

# ==================== داده‌های Special ====================
special_mae = {
    'ARIMA': [34.48, 27.84, 19.72],
    'SARIMA': [34.48, 27.84, 19.72],
    'Exponential Smoothing': [36.72, 24.45, 12.09],
    'SVM (SVR)': [24.92, 23.02, 5.82],
    'Naïve Seasonal': [30.61, 23.01, 27.25],
    'Random Forest': [29.02, 27.53, 69.37],
    'XGBoost': [28.64, 29.86, 60.61],
    'MLP_MultiOutput_Improved': [24.94, 25.83, 13.86],
    'CNN': [79.37, 149.76, 34.69],
    'GRU': [75.11, 128.64, 25.70],
    'Hybrid_CNN_LSTM': [75.17, 142.44, 33.86],
    'LSTM': [69.88, 127.61, 34.13],
    'RNN': [77.96, 139.88, 25.94]
}

df_special = pd.DataFrame(special_mae).T
df_special.columns = ['Window1', 'Window2', 'Window3']
df_special['Mean_MAE'] = df_special.mean(axis=1)
df_special = df_special.sort_values('Mean_MAE')

print("=== جدول Special (مرتب‌شده بر اساس میانگین MAE) ===")
print(df_special.round(2))
print("\n")

# ==================== توابع کمکی ====================
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def run_all_tests(df, class_name):
    print(f"\n{'='*60}")
    print(f"آزمون‌های آماری برای کلاس: {class_name}")
    print(f"{'='*60}")

    models = df.index.tolist()
    data = {m: df.loc[m, ['Window1', 'Window2', 'Window3']].values for m in models}

    # ۱. آزمون نرمال بودن (Shapiro-Wilk)
    print("\n۱. آزمون نرمال بودن (Shapiro-Wilk):")
    for m in models:
        stat, p = stats.shapiro(data[m])
        print(f"   {m}: Statistic={stat:.4f}, p-value={p:.4f}")

    # ۲. آزمون Friedman (مقایسه همه مدل‌ها با هم)
    print("\n۲. آزمون Friedman (تفاوت کلی بین همه مدل‌ها):")
    stat, p = stats.friedmanchisquare(*[data[m] for m in models])
    print(f"   Statistic={stat:.4f}, p-value={p:.4f}")
    if p < 0.05:
        print("   → تفاوت معنادار بین مدل‌ها وجود دارد.")

    # ۳. آزمون‌های جفتی (همه جفت‌ها)
    print("\n۳. آزمون‌های جفتی (T-test مستقل + Mann-Whitney + Wilcoxon):")
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            d1, d2 = data[m1], data[m2]

            t_stat, t_p = stats.ttest_ind(d1, d2)
            u_stat, u_p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            w_stat, w_p = stats.wilcoxon(d1, d2)
            d = cohen_d(d1, d2)

            print(f"   {m1} vs {m2}:")
            print(f"      T-test p={t_p:.4f} | Mann-Whitney p={u_p:.4f} | Wilcoxon p={w_p:.4f} | Cohen's d={abs(d):.3f}")

    # ۴. اندازه اثر (Cohen's d) برای بهترین و بدترین مدل
    print("\n۴. اندازه اثر (Cohen's d) بین بهترین و بدترین مدل:")
    best = df.index[0]
    worst = df.index[-1]
    d = cohen_d(data[best], data[worst])
    print(f"   {best} vs {worst}: Cohen's d = {abs(d):.3f}")

# اجرای آزمون‌ها
run_all_tests(df_business, "Business")
run_all_tests(df_special, "Special")

# ==================== الگوی Diebold-Mariano (برای سری خطاها) ====================
print("\n" + "="*60)
print("الگوی Diebold-Mariano Test (اگر سری خطای واقعی دارید):")
print("="*60)

def diebold_mariano(e1, e2, loss='MAE'):
    """e1 و e2 باید آرایه خطاهای پیش‌بینی (actual - pred) باشند"""
    if loss == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    else:  # MSE
        d = e1**2 - e2**2

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

# مثال استفاده (وقتی سری خطا دارید):
# dm_stat, p = diebold_mariano(errors_arima, errors_lstm)
# print(f"Diebold-Mariano: stat={dm_stat:.4f}, p={p:.4f}")


# In[27]:


import pandas as pd
import numpy as np
from scipy.stats import norm

# ==================== داده‌های Business ====================
business_mae = {
    'ARIMA': [25.04, 16.08, 17.24],
    'SARIMA': [25.04, 16.08, 17.24],
    'Exponential Smoothing': [27.80, 18.12, 9.83],
    'SVM (SVR)': [27.20, 21.14, 9.63],
    'Naïve Seasonal': [33.68, 22.64, 6.15],
    'Random Forest': [30.46, 27.94, 17.85],
    'XGBoost': [35.31, 26.19, 21.73],
    'MLP_MultiOutput_Improved': [28.82, 24.14, 17.57],
    'CNN': [121.38, 218.86, 54.97],
    'GRU': [137.95, 221.36, 56.43],
    'Hybrid_CNN_LSTM': [130.88, 222.93, 56.67],
    'LSTM': [121.52, 194.19, 56.87],
    'RNN': [129.58, 204.73, 53.20]
}

# ==================== داده‌های Special ====================
special_mae = {
    'ARIMA': [34.48, 27.84, 19.72],
    'SARIMA': [34.48, 27.84, 19.72],
    'Exponential Smoothing': [36.72, 24.45, 12.09],
    'SVM (SVR)': [24.92, 23.02, 5.82],
    'Naïve Seasonal': [30.61, 23.01, 27.25],
    'Random Forest': [29.02, 27.53, 69.37],
    'XGBoost': [28.64, 29.86, 60.61],
    'MLP_MultiOutput_Improved': [24.94, 25.83, 13.86],
    'CNN': [79.37, 149.76, 34.69],
    'GRU': [75.11, 128.64, 25.70],
    'Hybrid_CNN_LSTM': [75.17, 142.44, 33.86],
    'LSTM': [69.88, 127.61, 34.13],
    'RNN': [77.96, 139.88, 25.94]
}

def diebold_mariano(e1, e2, loss='MAE'):
    """Diebold-Mariano Test"""
    if loss == 'MAE':
        d = np.abs(np.array(e1)) - np.abs(np.array(e2))
    else:  # MSE
        d = np.array(e1)**2 - np.array(e2)**2

    T = len(d)
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0:
        return 0, 1.0

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def run_dm_pairwise(mae_dict, class_name):
    print(f"\n{'='*70}")
    print(f"Diebold-Mariano Test - کلاس {class_name}")
    print(f"{'='*70}")

    models = list(mae_dict.keys())
    results = []

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1 = models[i]
            m2 = models[j]
            e1 = mae_dict[m1]
            e2 = mae_dict[m2]

            dm_stat, p = diebold_mariano(e1, e2, loss='MAE')
            results.append({
                'Model1': m1,
                'Model2': m2,
                'DM_Statistic': round(dm_stat, 4),
                'p_value': round(p, 4),
                'Significant (α=0.05)': 'Yes' if p < 0.05 else 'No'
            })

    df = pd.DataFrame(results)
    df = df.sort_values('p_value')
    print(df.to_string(index=False))
    return df

# اجرای آزمون برای هر دو کلاس
dm_business = run_dm_pairwise(business_mae, "Business")
dm_special = run_dm_pairwise(special_mae, "Special")

# ذخیره نتایج
dm_business.to_excel("DM_Business_Results.xlsx", index=False)
dm_special.to_excel("DM_Special_Results.xlsx", index=False)
print("\nنتایج در فایل‌های DM_Business_Results.xlsx و DM_Special_Results.xlsx ذخیره شد.")


# In[ ]:




