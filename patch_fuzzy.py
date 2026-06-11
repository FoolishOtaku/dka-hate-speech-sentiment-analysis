# -*- coding: utf-8 -*-
"""
Patch the notebook fuzzy engine to fix low HS recall.

Changes:
  1. Add a 6th fuzzy input: f_dl_hs_prob (BiLSTM hate-speech probability) -> mf_dl_hs
  2. Replace the rule base with SPARSE covering rules (few antecedents each),
     so a single strong indicator can fire a high-severity rule instead of
     being ANDed-away by weak features (the root cause of the false negatives).
  3. Update FuzzyMamdani / FuzzySugeno / FuzzyMamdaniNoDL to consume the new input.
  4. Update the Mamdani-process visualisation cell to pass the new input.

The notebook is backed up to hate_speech_fuzzy_dl.ipynb.bak before writing.
Re-run the notebook from the BiLSTM-evaluation cell onward to get new numbers.
"""
import json, shutil, sys

SRC_NB = "_ours.ipynb"                       # clean HEAD version (your advanced work)
NB     = "hate_speech_fuzzy_dl_FIXED.ipynb"  # output: new, runnable, merge untouched
nb = json.load(open(SRC_NB, encoding="utf-8"))


def as_lines(src: str):
    # store source as a list of lines, keeping trailing newlines like Jupyter does
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def replace_cell(marker: str, new_src: str) -> bool:
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        joined = "".join(c["source"])
        if marker in joined:
            c["source"] = as_lines(new_src)
            c["outputs"] = []
            c["execution_count"] = None
            return True
    return False


# ────────────────────────────────────────────────────────────────────
# 1) MEMBERSHIP FUNCTIONS  (add mf_dl_hs for the BiLSTM HS probability)
# ────────────────────────────────────────────────────────────────────
MF_SRC = r'''# ═══════════════════════════════════════════════════════════════
# FUNGSI KEANGGOTAAN V3 — DIKALIBRASI + INPUT BiLSTM-HS
# Perbaikan utama dibanding V2:
#   - Menambah variabel ke-6: DL HS Probability (mf_dl_hs)
#     (probabilitas hate speech langsung dari BiLSTM — sinyal paling kuat,
#      sebelumnya dihitung tapi TIDAK dipakai oleh fuzzy)
# ═══════════════════════════════════════════════════════════════

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left  = (x - a) / (b - a + 1e-10)
    right = (c - x) / (c - b + 1e-10)
    return np.maximum(0, np.minimum(left, right))

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left   = (x - a) / (b - a + 1e-10)
    top    = np.ones_like(x)
    right  = (d - x) / (d - c + 1e-10)
    return np.maximum(0, np.minimum(np.minimum(left, top), right))

def gaussmf(x, mean, sigma):
    return np.exp(-((np.asarray(x, dtype=float) - mean)**2) / (2 * sigma**2))


# Variabel 1: Abusive Word Ratio
def mf_abusive_ratio(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.05,  0.15),
        'sedang' : trimf (x, 0.05, 0.20, 0.40),
        'tinggi' : trapmf(x, 0.25, 0.50, 1.0,   1.0)
    }

# Variabel 2: HS Keyword Score
def mf_hs_keyword(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.10,  0.25),
        'sedang' : trimf (x, 0.15, 0.33, 0.55),
        'tinggi' : trapmf(x, 0.45, 0.67, 1.0,   1.0)
    }

# Variabel 3: Tweet Negativity
def mf_negativity(x):
    return {
        'positif' : trapmf(x, 0,    0,    0.10,  0.25),
        'netral'  : trimf (x, 0.10, 0.33, 0.55),
        'negatif' : trapmf(x, 0.40, 0.60, 1.0,   1.0)
    }

# Variabel 4: Target Specificity
def mf_target(x):
    return {
        'umum'    : trapmf(x, 0,    0,    0.15,  0.35),
        'spesifik': trapmf(x, 0.25, 0.55, 1.0,   1.0)
    }

# Variabel 5: DL Abusive Probability (BiLSTM — output abusive)
def mf_dl_abusive(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.15,  0.35),
        'sedang' : trimf (x, 0.20, 0.50, 0.75),
        'tinggi' : trapmf(x, 0.60, 0.80, 1.0,   1.0)
    }

# Variabel 6 [BARU]: DL HS Probability (BiLSTM — output hate speech)
# Sinyal paling diskriminatif. BiLSTM cenderung confident -> MF tegas.
def mf_dl_hs(x):
    return {
        'rendah' : trapmf(x, 0,    0,    0.20,  0.40),
        'sedang' : trimf (x, 0.30, 0.55, 0.75),
        'tinggi' : trapmf(x, 0.60, 0.80, 1.0,   1.0)
    }

# Variabel Output: HS Severity
def mf_severity(x):
    return {
        'aman'    : trapmf(x, 0,    0,    0.12,  0.25),
        'lemah'   : trimf (x, 0.12, 0.33, 0.50),
        'sedang'  : trimf (x, 0.38, 0.55, 0.72),
        'kuat'    : trapmf(x, 0.60, 0.78, 1.0,   1.0)
    }

print('✅ MF V3 didefinisikan (6 input: + DL HS Probability)')
'''

# ────────────────────────────────────────────────────────────────────
# 2) RULE BASE  (sparse covering rules — break the 5-way MIN bottleneck)
# ────────────────────────────────────────────────────────────────────
RULES_SRC = r'''# ═══════════════════════════════════════════════════════════════
# RULE BASE V3 — SPARSE COVERING RULES (memperbaiki recall HS)
#
# MASALAH V2: setiap rule meng-AND-kan 5 input dengan MIN, sehingga
# firing strength = nilai terkecil dari 5 fitur. Karena f_abusive_ratio
# dan f_hs_keyword hampir selalu ~0, rule kelas "kuat"/"sedang" praktis
# TIDAK PERNAH menyala -> severity rendah -> banyak HS tidak terdeteksi.
#
# SOLUSI V3: rule dibuat SPARSE (antecedent sedikit, sisanya don't-care
# = None). Dengan begitu SATU indikator kuat saja sudah cukup menyalakan
# rule severity tinggi. Engine sudah mendukung None (lihat infer()).
#
# Kunci: f_dl_hs (probabilitas HS dari BiLSTM) dipakai sebagai pendorong
# utama, didukung kata-kunci HS, abusive ratio, negativity, dan target.
# Setiap dict hanya menulis antecedent yang relevan; infer() memakai .get
# sehingga key yang tidak ada otomatis dianggap don't-care.
# ═══════════════════════════════════════════════════════════════

rules = [
    # ── AMAN: BiLSTM yakin bukan HS DAN tidak ada sinyal abusif ──
    {'dlhs':'rendah', 'dl':'rendah',                       'out':'aman',  'w':1.00},
    {'dlhs':'rendah', 'neg':'positif',                     'out':'aman',  'w':0.85},
    {'ar':'rendah', 'hs':'rendah', 'dl':'rendah',          'out':'aman',  'w':0.80},

    # ── LEMAH: sinyal ringan / ambigu ──
    {'dl':'sedang',                                        'out':'lemah', 'w':0.90},
    {'ar':'sedang',                                        'out':'lemah', 'w':0.80},
    {'hs':'sedang',                                        'out':'lemah', 'w':0.80},
    {'dlhs':'sedang', 'dl':'rendah',                       'out':'lemah', 'w':0.75},

    # ── SEDANG: indikasi HS moderat ──
    {'dlhs':'sedang',                                      'out':'sedang','w':1.00},
    {'dl':'tinggi',                                        'out':'sedang','w':0.90},
    {'hs':'sedang', 'neg':'negatif',                       'out':'sedang','w':0.90},
    {'hs':'tinggi',                                        'out':'sedang','w':0.88},
    {'ar':'tinggi',                                        'out':'sedang','w':0.85},
    {'tgt':'spesifik', 'neg':'negatif',                    'out':'sedang','w':0.80},

    # ── KUAT: indikasi HS kuat ──
    {'dlhs':'tinggi',                                      'out':'kuat',  'w':1.00},
    {'dlhs':'sedang', 'tgt':'spesifik',                    'out':'kuat',  'w':0.95},
    {'dlhs':'sedang', 'dl':'tinggi',                       'out':'kuat',  'w':0.90},
    {'hs':'tinggi', 'neg':'negatif',                       'out':'kuat',  'w':0.92},
    {'hs':'tinggi', 'tgt':'spesifik',                      'out':'kuat',  'w':0.90},
    {'dl':'tinggi', 'tgt':'spesifik', 'neg':'negatif',     'out':'kuat',  'w':0.85},
]

print(f'✅ Rule base V3: {len(rules)} sparse rules')
print(f'{"No":>3}  {"antecedents":<48} -> {"output":8} w')
print('-'*72)
for i, r in enumerate(rules, 1):
    ante = {k: v for k, v in r.items() if k not in ("out", "w")}
    print(f"{i:>3}  {str(ante):<48} -> {r['out']:8} {r['w']}")
'''

# ────────────────────────────────────────────────────────────────────
# 3) MAMDANI  (6 inputs)
# ────────────────────────────────────────────────────────────────────
MAMDANI_SRC = r'''# ═══════════════════════════════════════════════════════════════
# FUZZY MAMDANI — V3 (6 input, sparse rules, don't-care via .get)
# Pipeline: Fuzzifikasi → Inferensi (MIN-MAX) → Defuzzifikasi (Centroid)
# Input: [abusive_ratio, hs_keyword, negativity, target_spec,
#         dl_abusive, dl_hs]
# ═══════════════════════════════════════════════════════════════

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

    def predict(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        mu          = self.fuzzify(ar, hs, neg, tgt, dl, dlhs)
        activations = self.infer(mu)
        aggregated  = self.aggregate(activations)
        return self.defuzzify(aggregated)

    def predict_batch(self, df_input):
        results = []
        for _, row in df_input.iterrows():
            results.append(self.predict(
                row['f_abusive_ratio'], row['f_hs_keyword'],
                row['f_negativity'],   row['f_target_spec'],
                row['f_dl_abusive'],   row['f_dl_hs_prob']
            ))
        return np.array(results)


fuzzy_mamdani = FuzzyMamdani()

sample = df.iloc[0]
score_test = fuzzy_mamdani.predict(
    sample['f_abusive_ratio'], sample['f_hs_keyword'],
    sample['f_negativity'],    sample['f_target_spec'],
    sample['f_dl_abusive'],    sample['f_dl_hs_prob']
)
print('=== Uji Sampel Mamdani (V3) ===')
print(f'Tweet  : {sample["Tweet"][:60]}...')
print(f'Label  : HS={sample["HS"]}, Abusive={sample["Abusive"]}')
print(f'Output : severity={score_test:.4f}')
'''

# ────────────────────────────────────────────────────────────────────
# 4) SUGENO  (6 inputs)
# ────────────────────────────────────────────────────────────────────
SUGENO_SRC = r'''# ═══════════════════════════════════════════════════════════════
# FUZZY SUGENO — V3 (zero-order, 6 input, sparse rules)
# Pipeline: Fuzzifikasi → Inferensi → Defuzzifikasi (Weighted Average)
# ═══════════════════════════════════════════════════════════════

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

    def predict(self, ar, hs, neg, tgt, dl, dlhs=0.0):
        mu          = self.fuzzify(ar, hs, neg, tgt, dl, dlhs)
        activations = self.infer(mu)
        return self.defuzzify(activations)

    def predict_batch(self, df_input):
        results = []
        for _, row in df_input.iterrows():
            results.append(self.predict(
                row['f_abusive_ratio'], row['f_hs_keyword'],
                row['f_negativity'],   row['f_target_spec'],
                row['f_dl_abusive'],   row['f_dl_hs_prob']
            ))
        return np.array(results)


fuzzy_sugeno = FuzzySugeno()
score_s = fuzzy_sugeno.predict(
    sample['f_abusive_ratio'], sample['f_hs_keyword'],
    sample['f_negativity'],    sample['f_target_spec'],
    sample['f_dl_abusive'],    sample['f_dl_hs_prob']
)
print('=== Uji Sampel Sugeno (V3) ===')
print(f'Output Mamdani : {score_test:.4f}')
print(f'Output Sugeno  : {score_s:.4f}')
'''

# ────────────────────────────────────────────────────────────────────
# 5) NO-DL baseline  (zero out BOTH DL inputs)
# ────────────────────────────────────────────────────────────────────
NODL_SRC = r'''# ═══════════════════════════════════════════════════════════════
# HYBRID SYSTEM: LSTM + FUZZY MAMDANI  (dampak DL pada fuzzy)
# Baseline tanpa DL  : kedua input BiLSTM (dl_abusive & dl_hs) = 0
# ═══════════════════════════════════════════════════════════════

class FuzzyMamdaniNoDL(FuzzyMamdani):
    """Mamdani tanpa fitur DL (dl=0 dan dlhs=0 selalu)."""
    def predict(self, ar, hs, neg, tgt, dl=0, dlhs=0):
        return super().predict(ar, hs, neg, tgt, 0.0, 0.0)

    def predict_batch(self, df_input):
        results = []
        for _, row in df_input.iterrows():
            results.append(self.predict(
                row['f_abusive_ratio'], row['f_hs_keyword'],
                row['f_negativity'],   row['f_target_spec'], 0.0, 0.0
            ))
        return np.array(results)

fuzzy_no_dl = FuzzyMamdaniNoDL()
print('Menjalankan Mamdani TANPA DL...')
df['mamdani_nodl_score']  = fuzzy_no_dl.predict_batch(df)
df['mamdani_nodl_hs_pred']= df['mamdani_nodl_score'].apply(score_to_hs)

acc_with_dl    = accuracy_score(hs, df['mamdani_hs_pred'])
acc_without_dl = accuracy_score(hs, df['mamdani_nodl_hs_pred'])
f1_with        = f1_score(hs, df['mamdani_hs_pred'],    average='weighted')
f1_without     = f1_score(hs, df['mamdani_nodl_hs_pred'],average='weighted')
auc_with       = roc_auc_score(hs, df['mamdani_score'])
auc_without    = roc_auc_score(hs, df['mamdani_nodl_score'])

print()
print('══════════════════════════════════════════════════════════')
print('Dampak Integrasi Deep Learning pada Sistem Fuzzy')
print('──────────────────────────────────────────────────────────')
print(f'{"Metrik":<20} {"Mamdani + DL":>18} {"Mamdani Only":>18} {"Δ":>8}')
print('──────────────────────────────────────────────────────────')
print(f'{"Accuracy":<20} {acc_with_dl:>18.4f} {acc_without_dl:>18.4f} {acc_with_dl-acc_without_dl:>+8.4f}')
print(f'{"F1-Score":<20} {f1_with:>18.4f} {f1_without:>18.4f} {f1_with-f1_without:>+8.4f}')
print(f'{"ROC-AUC":<20} {auc_with:>18.4f} {auc_without:>18.4f} {auc_with-auc_without:>+8.4f}')
print('══════════════════════════════════════════════════════════')
'''

patches = [
    ("FUNGSI KEANGGOTAAN V2", MF_SRC),
    ("RULE BASE V2", RULES_SRC),
    ("class FuzzyMamdani:", MAMDANI_SRC),
    ("class FuzzySugeno:", SUGENO_SRC),
    ("class FuzzyMamdaniNoDL", NODL_SRC),
]

results = []
for marker, src in patches:
    ok = replace_cell(marker, src)
    results.append((marker, ok))

# ── 6) Patch the Mamdani-process visualisation cell to pass dl_hs ──
viz_ok = False
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    joined = "".join(c["source"])
    if "Visualisasi Proses Mamdani secara Detail" in joined:
        joined = joined.replace(
            "DL  = hs_samples['f_dl_abusive']",
            "DL  = hs_samples['f_dl_abusive']\nDLHS= hs_samples['f_dl_hs_prob']"
        )
        joined = joined.replace(
            "mu = fuzzy_mamdani.fuzzify(AR, HS_, NEG, TGT, DL)",
            "mu = fuzzy_mamdani.fuzzify(AR, HS_, NEG, TGT, DL, DLHS)"
        )
        c["source"] = as_lines(joined)
        c["outputs"] = []
        c["execution_count"] = None
        viz_ok = True
        break
results.append(("mamdani_process viz", viz_ok))

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("Patched notebook:", NB)
for marker, ok in results:
    print(f"  [{'OK' if ok else 'MISS'}] {marker}")
if not all(ok for _, ok in results):
    sys.exit("Some cells were not found — check markers.")
