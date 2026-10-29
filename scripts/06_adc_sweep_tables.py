# ============================================================================
#  TABLO 2 (ADC sweep) + TABLO 3 (variation) — yeni tutarli kosum
#  --------------------------------------------------------------------------
#  Amac: Tablo 2/3'u, SAA/cross-dataset ile AYNI ayarla yeniden uret ki
#  makalede celiski kalmasin.
#
#  Tablo 2: 4 veri seti x ADC {5,8,12} uniform  (--ADCprecision)
#  Tablo 3: Pneumonia, ADC {5,8} x vari {0,0.05,0.1,0.2}  (--ADCprecision --vari)
#
#  Runtime: T4 GPU. Kosum basina Pneumonia/Retina ~10 dk, Derma ~30 dk,
#  Blood ~50 dk. Toplam ~3-4 saat. Ara-kayit + devam destekli.
#  NOT: CIFAR-10 satirlari (Tablo 2) bu hucrede YOK — onlar ayri; mevcut
#  degerleri korunacak (AUC zaten '-' idi).
# ============================================================================

import subprocess, sys, os, re, csv, time
try:
    import medmnist
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","medmnist"]); import medmnist
from medmnist import INFO
from google.colab import drive
drive.mount('/content/drive')

REPO = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/Inference_pytorch"
OUT2 = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures/table2_sweep_NEW.csv"
OUT3 = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures/table3_variation_NEW.csv"
os.chdir(REPO)

BASE = ["--model","VGG8","--mode","WAGE","--inference","1",
        "--subArray","128","--parallelRead","128","--seed","117","--skip_hw_estimate","1"]

def parse(text):
    acc=None; m=re.findall(r'Accuracy:\s*([0-9]+)\s*/\s*([0-9]+)',text)
    if m: c,t=m[-1]; acc=round(100*int(c)/int(t),2)
    bacc=None; m=re.findall(r'Balanced\s*Accuracy:\s*([0-9]+\.?[0-9]*)',text,re.I)
    if m: bacc=float(m[-1])
    auc=None; m=re.findall(r'\bAUC:\s*([0-9]*\.?[0-9]+)',text,re.I)
    if m: auc=float(m[-1])
    return acc,bacc,auc

def run(extra, tag, timeout=5400):
    cmd=["python","inference.py"]+extra+BASE
    print(f"\n{'='*55}\n{tag}\n  $ {' '.join(cmd)}")
    t0=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout); log=r.stdout+"\n"+r.stderr
    except Exception as e: log=f"[HATA] {e}"
    acc,bacc,auc=parse(log); dt=time.time()-t0
    for ln in log.splitlines():
        if any(k in ln for k in ["Accuracy:","Balanced","AUC:"]): print("   ",ln.strip())
    print(f"   --> acc={acc} bacc={bacc} auc={auc} ({dt:.0f}s)")
    return acc,bacc,auc

# ---------- TABLO 2: ADC sweep ----------
datasets=["pneumoniamnist","dermamnist","bloodmnist","retinamnist"]
sweep_bits=[5,8,12]
# devam: mevcut CSV'yi oku
def load(path):
    rows=[]
    if os.path.exists(path):
        with open(path) as f: rows=list(csv.DictReader(f))
    return rows
def save(path, rows, fields):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

rows2=load(OUT2)
done2={(r["dataset"],r["adc_bits"]) for r in rows2 if r.get("auc") not in (None,"","None")}
for ds in datasets:
    for b in sweep_bits:
        if (ds,str(b)) in done2: print(f"[ATLA] T2 {ds} {b}bit"); continue
        acc,bacc,auc=run(["--dataset",ds,"--ADCprecision",str(b)], f"T2 {ds} ADC={b}")
        rows2=[r for r in rows2 if not (r["dataset"]==ds and r["adc_bits"]==str(b))]
        rows2.append({"dataset":ds,"adc_bits":str(b),"accuracy_pct":acc,
                      "balanced_accuracy_pct":bacc,"auc":auc})
        save(OUT2, rows2, ["dataset","adc_bits","accuracy_pct","balanced_accuracy_pct","auc"])

# ---------- TABLO 3: variation (Pneumonia) ----------
rows3=load(OUT3)
done3={(r["adc_bits"],r["vari"]) for r in rows3 if r.get("auc") not in (None,"","None")}
for b in [5,8]:
    for v in [0.0,0.05,0.10,0.20]:
        if (str(b),str(v)) in done3: print(f"[ATLA] T3 {b}bit vari={v}"); continue
        acc,bacc,auc=run(["--dataset","pneumoniamnist","--ADCprecision",str(b),"--vari",str(v)],
                         f"T3 ADC={b} vari={v}")
        rows3=[r for r in rows3 if not (r["adc_bits"]==str(b) and r["vari"]==str(v))]
        rows3.append({"adc_bits":str(b),"vari":str(v),"accuracy_pct":acc,
                      "balanced_accuracy_pct":bacc,"auc":auc})
        save(OUT3, rows3, ["adc_bits","vari","accuracy_pct","balanced_accuracy_pct","auc"])

print("\n"+"="*55+"\nTAMAMLANDI")
print("Tablo2:",OUT2,"\nTablo3:",OUT3)
import pandas as pd
print("\n--- TABLO 2 (sweep) ---"); print(pd.DataFrame(rows2).to_string(index=False))
print("\n--- TABLO 3 (variation) ---"); print(pd.DataFrame(rows3).to_string(index=False))
