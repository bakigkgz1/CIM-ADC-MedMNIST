# ============================================================================
#  SENSITIVITY-DRIVEN AUTOMATIC ADC ALLOCATION  (SAA)
#  --------------------------------------------------------------------------
#  Amac: Katman-bazli ADC bit-genisligini ELLE secmek yerine, leave-one-low
#  ablation duyarliligindan OTOMATIK ureten ilkesel bir algoritma. Boylece
#  makale "gozlem" olmaktan cikip "yontem" katkisi kazanir.
#
#  Fikir:
#   1) Her katmanin 5-bit'e dusurulunce verdigi AUC kaybi = DUYARLILIK.
#   2) Katmanlari duyarliliga gore sirala (en dayanikli -> en kirilgan).
#   3) Hedef bir ortalama-bit butcesi B verildiginde, en dayanikli katmanlari
#      sirayla dusuk-bite indir (greedy) -> B'yi saglayan tahsis.
#   4) B'yi tarayarak tum dogruluk-maliyet PARETO cephesini otomatik uret.
#
#  Girdi : table3 (layer ablation) CSV  (Drive, paper_figures klasoru)
#  Cikti : SAA_allocation_results.csv  (uretilen tahsis ailesi)
#
#  Yaklasik calisma suresi: < 1 dakika. Bu hucre YENI bir ag egitimi veya
#  NeuroSim kosumu GEREKTIRMEZ; sadece mevcut ablation verisi uzerinde
#  hesaplama yapar. Algoritmanin onerdigi YENI tahsisleri (MIX_C disindakiler)
#  gercek metrikle dogrulamak isterseniz, HUCRE 6'daki listeyi mevcut
#  inference pipeline'inizda kosun (her tahsis ~tek inference kosumu).
# ============================================================================

# ---- HUCRE 1: Drive ----
from google.colab import drive
drive.mount('/content/drive')

# ---- HUCRE 2: Ayarlar ----
import pandas as pd, numpy as np, io, os

BASE = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures"
T3 = os.path.join(BASE, "table3_layer_ablation_pneumonia.csv")

LAYER_NAMES = ["features.0","features.2","features.5","features.7",
               "features.10","features.12","classifier.0","classifier.2"]
LOW, MID, HIGH = 5, 8, 12     # aday bit-genislikleri

# ---- HUCRE 3: Ablation oku, duyarlilik cikar, katmanlari sirala ----
def read_csv_skip_comments(path):
    with open(path) as f:
        lines = [l for l in f if not l.lstrip().startswith('#')]
    return pd.read_csv(io.StringIO(''.join(lines)))

t3 = read_csv_skip_comments(T3)
all12 = t3[t3.config=='ALL12'].iloc[0]
base_auc, base_bacc = float(all12.auc), float(all12.balanced_accuracy_pct)

drop = {}
for i, name in enumerate(LAYER_NAMES):
    row = t3[t3.config==f"LOW{i}_{name}"].iloc[0]
    drop[i] = (base_auc - float(row.auc), base_bacc - float(row.balanced_accuracy_pct))

order = sorted(range(len(LAYER_NAMES)), key=lambda i: drop[i][0])  # dayanikli -> kirilgan
print("=== Katman dayaniklilik siralamasi (5-bit AUC kaybi) ===")
for i in order:
    print(f"  L{i} {LAYER_NAMES[i]:14s} auc_drop={drop[i][0]:+.4f}  bacc_drop={drop[i][1]:+.2f}")

# ---- HUCRE 4: Butce-tabanli otomatik tahsis (SAA) ----
def allocate_for_budget(B):
    """Hedef ortalama-bit B icin: en dayanikli katmanlari sirayla dusur."""
    bits = [HIGH]*len(LAYER_NAMES)
    # once en dayanikli katmanlari LOW'a cek
    for i in order:
        if np.mean(bits) <= B: break
        bits[i] = LOW
    # ince ayar: hedefin cok altina indiyse, en son dusurulen katmani MID'e cikar
    if np.mean(bits) < B - 0.5:
        for i in reversed(order):
            if bits[i]==LOW:
                bits[i]=MID
                if np.mean(bits)>=B-0.25: break
    return bits

budgets = [11.5, 11.0, 10.5, 10.25, 10.0, 9.5, 9.0]
family, seen = [], set()
for B in budgets:
    a = allocate_for_budget(B)
    k = tuple(a)
    if k in seen: continue
    seen.add(k)
    family.append({"policy":f"SAA(B<={B})","alloc":a,"avg_bits":round(np.mean(a),3)})

print("\n=== SAA'nin otomatik urettigi tahsis ailesi (Pareto) ===")
for p in family:
    print(f"  {p['policy']:14s} {p['alloc']} avg={p['avg_bits']}")

# ---- HUCRE 5: Elle secilen MIX_C ile karsilastir + kaydet ----
mixc = [5,12,12,12,12,12,12,5]
reproduced = any(p["alloc"]==mixc for p in family)
print(f"\nElle secilen MIX_C: {mixc} avg={np.mean(mixc):.3f}")
print(f"--> SAA bunu OTOMATIK uretiyor mu? {reproduced}")
print("    (Evet ise: elle bulunan cozum, ilkesel algoritmanin ozel halidir.)")

out = pd.DataFrame([{"policy":p["policy"],
                     "adc_per_layer":",".join(map(str,p["alloc"])),
                     "avg_bits":p["avg_bits"]} for p in family])
out.loc[len(out)] = ["MIX_C (manual)", ",".join(map(str,mixc)), round(np.mean(mixc),3)]
save = os.path.join(BASE, "SAA_allocation_results.csv")
out.to_csv(save, index=False)
print(f"\nKaydedildi: {save}")
print(out.to_string(index=False))

# ---- HUCRE 6 (OPSIYONEL): Yeni tahsisleri NeuroSim'de dogrula ----
new = [p["alloc"] for p in family if p["alloc"]!=mixc]
print("\n=== NeuroSim ile dogrulanacak YENI tahsisler (MIX_C disindakiler) ===")
for a in new:
    print("  --ADC_per_layer", ",".join(map(str,a)))
print("Her biri ~tek inference kosumu; mevcut inference.py pipeline'inizla kosun.")
print("Olcilen AUC/acc'leri tabloya ekleyince SAA Pareto cephesi tamamlanir.")
