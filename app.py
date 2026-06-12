import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import re
import warnings

# Suppress tensorflow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set Streamlit page layout
st.set_page_config(
    page_title="Deteksi Hate Speech Hybrid",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #f9fbfd;
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ──
kamusalay_path = 'data/new_kamusalay.csv'
abusive_path = 'data/abusive.csv'
model_path = 'saved_models/bilstm_multioutput.keras'
tokenizer_path = 'saved_models/tokenizer.pkl'
config_path = 'saved_models/config.pkl'

@st.cache_resource
def load_resources():
    # Load kamus alay
    df_kamus = pd.read_csv(kamusalay_path, encoding='latin1', header=0)
    df_kamus.columns = ['alay', 'normal']
    alay_dict = dict(zip(df_kamus['alay'].str.lower(), df_kamus['normal'].str.lower()))

    # Load abusive lexicon
    df_abuse = pd.read_csv(abusive_path, encoding='utf-8')
    abusive_set = set(df_abuse['ABUSIVE'].str.lower().str.strip())

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load config
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Load LSTM model
    model = tf.keras.models.load_model(model_path)

    return alay_dict, abusive_set, tokenizer, config, model

# Load all resources
try:
    alay_dict, abusive_set, tokenizer, config, model_lstm = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource model atau data: {e}")
    st.stop()

# ── Preprocessing & Features ──
def normalize_alay(text):
    words = text.split()
    return ' '.join(alay_dict.get(w, w) for w in words)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\buser\b|\brt\b|\burl\b', '', text)  # token khusus
    text = re.sub(r'http\S+|www\S+', '', text)             # URL
    text = re.sub(r'[^a-z0-9\s]', ' ', text)               # punctuation
    text = normalize_alay(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

hs_keywords = set([
    'kafir', 'murtad', 'sesat', 'laknat', 'haram', 'terkutuk', 'munafik', 'musyrik',
    'babi', 'celeng', 'anjing', 'monyet', 'kunyuk', 'bangkai', 'binatang',
    'komunis', 'teroris', 'cebong', 'kampret', 'antek', 'pki',
    'pribumi', 'pendatang', 'sipit', 'bong', 'asing',
    'sampah', 'najis', 'nista', 'hina', 'rendah', 'jijik',
    'usir', 'bunuh', 'hancur', 'binasa', 'musnah', 'singkir',
    'iblis', 'setan', 'terkutuk', 'brengsek', 'keparat', 'bajingan', 'bangsat',
])

negation_words  = ['tidak','bukan','gak','ga','nggak','enggak','tak','jangan',
                   'tiada','tanpa','anti','lawan']
intensifier_neg = ['sangat','banget','sekali','amat','betul','memang','sungguh',
                   'benar','parah','habis','abis','mati']

def get_abusive_ratio(text):
    words = text.split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in abusive_set)
    return count / len(words)

def get_hs_keyword_score(text):
    words = set(text.split())
    hits  = sum(1 for kw in hs_keywords if kw in words)
    return min(hits / 3.0, 1.0) 

def get_tweet_negativity(text):
    words = text.split()
    abv_count = sum(1 for w in words if w in abusive_set)
    neg       = sum(1 for w in words if w in negation_words)
    inten     = sum(1 for w in words if w in intensifier_neg)
    raw       = (abv_count * 0.5 + neg * 0.3 + inten * 0.2)
    return min(raw / 3.0, 1.0)

def calculate_target_specificity(individual, group, religion, race):
    score = (individual * 0.4 + group * 0.3 + religion * 0.15 + race * 0.15)
    return float(score)

# ── Fuzzy Membership Functions ──
def trimf(x, a, b, c):
    x_arr = np.asarray(x, dtype=float)
    left = np.ones_like(x_arr) if a == b else (x_arr - a) / (b - a)
    right = np.ones_like(x_arr) if b == c else (c - x_arr) / (c - b)
    res = np.minimum(left, right)
    return np.maximum(0.0, res)

def trapmf(x, a, b, c, d):
    x_arr = np.asarray(x, dtype=float)
    left = np.ones_like(x_arr) if a == b else (x_arr - a) / (b - a)
    right = np.ones_like(x_arr) if c == d else (d - x_arr) / (d - c)
    res = np.minimum(np.minimum(left, 1.0), right)
    return np.maximum(0.0, res)

def mf_abusive_ratio(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.05,  0.15),
        'sedang' : trimf (x, 0.05, 0.20, 0.40),
        'tinggi' : trapmf(x, 0.25, 0.50, 1.0,   1.0)
    }

def mf_hs_keyword(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.10,  0.25),
        'sedang' : trimf (x, 0.15, 0.33, 0.55),
        'tinggi' : trapmf(x, 0.45, 0.67, 1.0,   1.0)
    }

def mf_negativity(x):
    return {
        'positif' : trapmf(x, 0,    0,    0.10,  0.25),
        'netral'  : trimf (x, 0.10, 0.33, 0.55),
        'negatif' : trapmf(x, 0.40, 0.60, 1.0,   1.0)
    }

def mf_target(x):
    return {
        'umum'    : trapmf(x, 0,    0,    0.15,  0.35),
        'spesifik': trapmf(x, 0.25, 0.55, 1.0,   1.0)
    }

def mf_dl_abusive(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.15,  0.35),
        'sedang' : trimf (x, 0.20, 0.50, 0.75),
        'tinggi' : trapmf(x, 0.60, 0.80, 1.0,   1.0)
    }

def mf_dl_hs(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.20,  0.40),
        'sedang' : trimf (x, 0.30, 0.55, 0.75),
        'tinggi' : trapmf(x, 0.60, 0.80, 1.0,   1.0)
    }

def mf_severity(x):
    return {
        'aman'    : trapmf(x, 0,    0,    0.12,  0.25),
        'lemah'   : trimf (x, 0.12, 0.33, 0.50),
        'sedang'  : trimf (x, 0.38, 0.55, 0.72),
        'kuat'    : trapmf(x, 0.60, 0.78, 1.0,   1.0)
    }

rules = [
    {'dlhs':'rendah', 'dl':'rendah',                       'out':'aman',  'w':1.00},
    {'dlhs':'rendah', 'neg':'positif',                     'out':'aman',  'w':0.85},
    {'ar':'rendah', 'hs':'rendah', 'dl':'rendah',          'out':'aman',  'w':0.80},
    {'dl':'sedang',                                        'out':'lemah', 'w':0.90},
    {'ar':'sedang',                                        'out':'lemah', 'w':0.80},
    {'hs':'sedang',                                        'out':'lemah', 'w':0.80},
    {'dlhs':'sedang', 'dl':'rendah',                       'out':'lemah', 'w':0.75},
    {'dlhs':'sedang',                                      'out':'sedang','w':1.00},
    {'dl':'tinggi',                                        'out':'sedang','w':0.90},
    {'hs':'sedang', 'neg':'negatif',                       'out':'sedang','w':0.90},
    {'hs':'tinggi',                                        'out':'sedang','w':0.88},
    {'ar':'tinggi',                                        'out':'sedang','w':0.85},
    {'tgt':'spesifik', 'neg':'negatif',                    'out':'sedang','w':0.80},
    {'dlhs':'tinggi',                                      'out':'kuat',  'w':1.00},
    {'dlhs':'sedang', 'tgt':'spesifik',                    'out':'kuat',  'w':0.95},
    {'dlhs':'sedang', 'dl':'tinggi',                       'out':'kuat',  'w':0.90},
    {'hs':'tinggi', 'neg':'negatif',                       'out':'kuat',  'w':0.92},
    {'hs':'tinggi', 'tgt':'spesifik',                      'out':'kuat',  'w':0.90},
    {'dl':'tinggi', 'tgt':'spesifik', 'neg':'negatif',     'out':'kuat',  'w':0.85},
]

class FuzzyMamdani:
    def __init__(self):
        self.output_range = np.linspace(0, 1, 1000)

    def fuzzify(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        return {
            'ar'  : mf_abusive_ratio(ar),
            'hs'  : mf_hs_keyword(hs),
            'neg' : mf_negativity(neg),
            'tgt' : mf_target(tgt),
            'dl'  : mf_dl_abusive(dl),
            'dlhs': mf_dl_hs(dlhs),
        }

    def infer(self, mu):
        rule_activations = []
        for rule in rules:
            conditions = []
            for var in ('ar', 'hs', 'neg', 'tgt', 'dl', 'dlhs'):
                label = rule.get(var)
                if label is not None:
                    conditions.append(mu[var].get(label, 0))
            firing = min(conditions) * rule['w'] if conditions else 0
            rule_activations.append((firing, rule['out']))
        return rule_activations

    def aggregate(self, rule_activations):
        x = self.output_range
        aggregated = np.zeros_like(x)
        for firing, out_label in rule_activations:
            if firing <= 0:
                continue
            out_mf  = mf_severity(x)
            clipped = np.minimum(out_mf[out_label], firing)
            aggregated = np.maximum(aggregated, clipped)
        return aggregated

    def defuzzify(self, aggregated):
        x = self.output_range
        denom = np.sum(aggregated)
        if denom < 1e-10:
            return 0.0
        return float(np.sum(x * aggregated) / denom)

    def predict_detailed(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        mu          = self.fuzzify(ar, hs, neg, tgt, dl, dlhs)
        activations = self.infer(mu)
        aggregated  = self.aggregate(activations)
        score       = self.defuzzify(aggregated)
        return score, mu, activations, aggregated

SUGENO_OUTPUTS = {
    'aman'  : 0.08,
    'lemah' : 0.35,
    'sedang': 0.62,
    'kuat'  : 0.92
}

class FuzzySugeno:
    def fuzzify(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        return {
            'ar'  : mf_abusive_ratio(ar),
            'hs'  : mf_hs_keyword(hs),
            'neg' : mf_negativity(neg),
            'tgt' : mf_target(tgt),
            'dl'  : mf_dl_abusive(dl),
            'dlhs': mf_dl_hs(dlhs),
        }

    def infer(self, mu):
        activations = []
        for rule in rules:
            conditions = []
            for var in ('ar', 'hs', 'neg', 'tgt', 'dl', 'dlhs'):
                label = rule.get(var)
                if label is not None:
                    conditions.append(mu[var].get(label, 0))
            firing    = min(conditions) * rule['w'] if conditions else 0
            const_out = SUGENO_OUTPUTS[rule['out']]
            activations.append((firing, const_out))
        return activations

    def defuzzify(self, activations):
        weighted_sum = sum(f * z for f, z in activations)
        total_weight = sum(f     for f, _ in activations)
        if total_weight < 1e-10:
            return 0.0
        return float(weighted_sum / total_weight)

    def predict_detailed(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        mu          = self.fuzzify(ar, hs, neg, tgt, dl, dlhs)
        activations = self.infer(mu)
        score       = self.defuzzify(activations)
        return score, mu, activations

# ── Sidebar Info ──
st.sidebar.markdown(f"""
<div style="text-align: center; padding-bottom: 20px;">
    <h2 style="margin: 0; color: #2196F3;">Deteksi Hate Speech</h2>
    <p style="font-size: 0.9rem; color: #666;">Hybrid BiLSTM & Logika Fuzzy</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("""
**Tentang Aplikasi:**
Aplikasi ini mendeteksi ujaran kebencian (*hate speech*) dan kata-kata abusif pada Tweet bahasa Indonesia menggunakan arsitektur hibrida.
- **Deep Learning (BiLSTM)** memprediksi probabilitas konten abusif dan ujaran kebencian.
- **Linguistic Features** mengekstrak rasio kata kasar, kata kunci HS, negativitas, dan kepesifikasian target.
- **Fuzzy Inference System (FIS)** Mamdani & Sugeno menentukan keputusan akhir.
""")

st.sidebar.markdown("### Kamus Kata Kunci HS")
with st.sidebar.expander("Lihat Kata Kunci HS Relevan"):
    st.write(", ".join(sorted(list(hs_keywords))))

# ── Main Content Header ──
st.markdown("""
<div style="background-color: #2196F3; padding: 25px; border-radius: 12px; margin-bottom: 30px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: bold; color: white;">Dashboard Analisis Hate Speech Hibrida</h1>
    <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9;">Integrasi BiLSTM Deep Learning & Logika Fuzzy Mamdani/Sugeno</p>
</div>
""", unsafe_allow_html=True)

# Main Grid Layout
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("### Input Tweet")
    tweet_input = st.text_area(
        "Masukkan teks tweet untuk diuji:",
        value="",
        height=100
    )
    
    st.markdown("### Parameter Target Ujaran Kebencian")
    st.markdown("<p style='font-size: 0.85rem; color: #555; margin-top: -10px;'>Pilih salah satu atau lebih target spesifik jika tweet tersebut menargetkan entitas berikut:</p>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        indiv = st.checkbox("Menargetkan Individu", value=False)
        religion = st.checkbox("Menargetkan Isu Agama", value=False)
    with col_t2:
        group = st.checkbox("Menargetkan Kelompok/Golongan", value=False)
        race = st.checkbox("Menargetkan Isu Ras/Etnis", value=False)
        
    tgt_score = calculate_target_specificity(
        1.0 if indiv else 0.0,
        1.0 if group else 0.0,
        1.0 if religion else 0.0,
        1.0 if race else 0.0
    )
    
    analyze_btn = st.button("Analisis Tweet", type="primary", use_container_width=True)

if analyze_btn or tweet_input:
    # 1. Preprocess
    clean_tweet = preprocess_text(tweet_input)
    
    # 2. Extract Manual Features
    ar_score = get_abusive_ratio(clean_tweet)
    hs_score = get_hs_keyword_score(clean_tweet)
    neg_score = get_tweet_negativity(clean_tweet)
    
    # 3. Predict Deep Learning
    seq = tokenizer.texts_to_sequences([clean_tweet])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=config['MAX_LEN'], padding='post', truncating='post')
    
    preds = model_lstm.predict(padded, verbose=0)
    dl_abusive = float(preds[0][0][0])
    dl_hs_prob = float(preds[1][0][0])
    
    # 4. Fuzzy Engine Inference
    fuzzy_mamdani = FuzzyMamdani()
    fuzzy_sugeno = FuzzySugeno()
    
    score_m, mu_m, activations_m, aggregated_m = fuzzy_mamdani.predict_detailed(
        ar_score, hs_score, neg_score, tgt_score, dl_abusive, dl_hs_prob
    )
    score_s, mu_s, activations_s = fuzzy_sugeno.predict_detailed(
        ar_score, hs_score, neg_score, tgt_score, dl_abusive, dl_hs_prob
    )
    
    # Display results on the right column
    with col_right:
        st.markdown("### Ringkasan Deteksi")
        
        # Binary Classification Result Alert (using optimal thresholds from notebook: Mamdani = 0.4715, Sugeno = 0.5242)
        is_mamdani_hs = score_m >= 0.4715
        is_sugeno_hs = score_s >= 0.5242
        
        if is_mamdani_hs or is_sugeno_hs:
            st.error("**Kalimat terdeteksi sebagai Ujaran Kebencian (Hate Speech).**")
        else:
            st.success("**Kalimat diklasifikasikan sebagai Aman (Bukan Hate Speech).**")
            
        st.write(f"**Teks Hasil Preprocessing:** `{clean_tweet}`")
        
        # Metric Grid
        m1, m2, m3 = st.columns(3)
        m1.metric("BiLSTM HS Score", f"{dl_hs_prob:.4f}")
        m2.metric("BiLSTM Abusive Score", f"{dl_abusive:.4f}")
        m3.metric("Linguistic Abusive Ratio", f"{ar_score:.4f}")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Linguistic Negativity", f"{neg_score:.4f}")
        m5.metric("Linguistic Keyword Score", f"{hs_score:.4f}")
        m6.metric("Target Specificity", f"{tgt_score:.4f}")
        
        # Fuzzy Mamdani & Sugeno side-by-side results
        st.markdown("### Output Sistem Fuzzy")
        col_m, col_s = st.columns(2)
        
        with col_m:
            st.subheader("Fuzzy Mamdani (Centroid)")
            st.metric("Severity Score", f"{score_m:.4f}")
            if score_m < 0.25:
                st.info("Kategori: **Aman**")
            elif score_m < 0.50:
                st.warning("Kategori: **Hate Speech Lemah**")
            elif score_m < 0.75:
                st.error("Kategori: **Hate Speech Sedang**")
            else:
                st.error("Kategori: **Hate Speech Kuat**")
            
            # Keputusan optimal threshold (Mamdani = 0.4715)
            if score_m >= 0.4715:
                st.markdown("Keputusan: **Hate Speech (HS)**")
            else:
                st.markdown("Keputusan: **Bukan Hate Speech (Non-HS)**")
                
        with col_s:
            st.subheader("Fuzzy Sugeno (W.Avg)")
            st.metric("Severity Score", f"{score_s:.4f}")
            if score_s < 0.25:
                st.info("Kategori: **Aman**")
            elif score_s < 0.50:
                st.warning("Kategori: **Hate Speech Lemah**")
            elif score_s < 0.75:
                st.error("Kategori: **Hate Speech Sedang**")
            else:
                st.error("Kategori: **Hate Speech Kuat**")
            
            # Keputusan optimal threshold (Sugeno = 0.5242)
            if score_s >= 0.5242:
                st.markdown("Keputusan: **Hate Speech (HS)**")
            else:
                st.markdown("Keputusan: **Bukan Hate Speech (Non-HS)**")

    # ── Full Width Visualizations ──
    st.markdown("---")
    st.markdown("### Visualisasi Proses Fuzzifikasi, Inferensi & Defuzzifikasi")
    
    # Generate Plots
    x_out = np.linspace(0, 1, 1000)
    out_mf = mf_severity(x_out)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors_in = ['#F44336','#FF9800','#2196F3','#9C27B0','#00BCD4']
    colors_out = {'aman':'#4CAF50','lemah':'#8BC34A','sedang':'#FF9800','kuat':'#F44336'}
    
    # Subplot 1: Nilai Input Variabel Fuzzy
    ax0 = axes[0]
    labels_in = ['Abusive\nRatio', 'HS\nKeyword', 'Negativity', 'Target\nSpec', 'DL\nProb']
    values_in = [ar_score, hs_score, neg_score, tgt_score, dl_hs_prob]
    bars = ax0.bar(labels_in, values_in, color=colors_in, width=0.5, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values_in):
        ax0.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax0.set_title('Nilai Input Variabel Fuzzy', fontweight='bold', fontsize=11)
    ax0.set_ylabel('Nilai Input (0-1)')
    ax0.set_ylim(0, 1.2)
    ax0.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    ax0.grid(True, alpha=0.2)
    
    # Subplot 2: Defuzzifikasi Mamdani
    ax1 = axes[1]
    for lbl, y in out_mf.items():
        ax1.plot(x_out, y, label=lbl, color=colors_out[lbl], linewidth=1.5, linestyle='--', alpha=0.6)
    ax1.fill_between(x_out, aggregated_m, alpha=0.35, color='#9C27B0', label='Area Agregasi')
    ax1.plot(x_out, aggregated_m, color='#9C27B0', linewidth=2)
    ax1.axvline(score_m, color='red', linewidth=2.5, linestyle='-.', label=f'CoG={score_m:.3f}')
    ax1.set_title('Defuzzifikasi Mamdani (Centroid)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Severity Score')
    ax1.set_ylabel('μ(x)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.2)
    
    # Subplot 3: Defuzzifikasi Sugeno
    ax2 = axes[2]
    agg_sugeno = {}
    for firing, const_out in activations_s:
        agg_sugeno[const_out] = max(agg_sugeno.get(const_out, 0), firing)
        
    for label, x_val in SUGENO_OUTPUTS.items():
        height = agg_sugeno.get(x_val, 0.0)
        ax2.vlines(x_val, 0, height, color=colors_out[label], linewidth=3, label=f'{label} (x={x_val})')
        ax2.plot(x_val, height, 'o', color=colors_out[label], markersize=8)
        ax2.text(x_val, height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax2.axvline(score_s, color='blue', linewidth=2.5, linestyle='-.', label=f'W.Avg={score_s:.3f}')
    ax2.set_title('Defuzzifikasi Sugeno (Weighted Average)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Severity Score')
    ax2.set_ylabel('Firing Strength (μ)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ── Explanations ──
    st.info("""
    **Catatan Interpretasi Grafik:**
    1. **Grafik Kiri (Input)** menunjukkan nilai masukan crisp untuk masing-masing variabel masukan logika fuzzy.
    2. **Grafik Tengah (Mamdani)** mengilustrasikan daerah agregasi fuzzy (warna ungu) hasil evaluasi rule base. Garis merah (`CoG`) mewakili titik tengah centroid sebagai output defuzzifikasi.
    3. **Grafik Kanan (Sugeno)** mengilustrasikan output model Sugeno di mana masing-masing rule menghasilkan singleton konstan. Garis biru (`W.Avg`) mewakili nilai rata-rata terbobot dari kekuatan aktivasi aturan tersebut.
    """)
