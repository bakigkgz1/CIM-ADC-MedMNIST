# ============================================================================
#  CROSS-DATASET SAA DOGRULAMA
#  --------------------------------------------------------------------------
#  Pneumonia'da turetilen SAA tahsislerini DIGER uc veri setinde (Derma, Blood,
#  Retina) YENIDEN TUNE ETMEDEN kos. Amac: SAA tahsisinin gorevler arasi
#  transfer oldugunu / genellenebildigini gostermek.
#
#  Her veri seti icin: ALL12 (ust referans), ALL5 (alt referans),
#  MIX_C, MIX_E, SAA_B11.0 (Pneumonia'da turetilen tahsisler).
#
#  Runtime: T4 GPU.  Sure: 4 veri seti x 5 tahsis x ~10 dk ≈ 3-3.5 saat.
#  ** Uzun! ** Colab oturum limitine dikkat. Ara-kayit var; oturum duserse
#  kaldigi yerden DEVAM etmek icin asagidaki DONE_SKIP mantigi mevcut kayitli
#  satirlari atlar.
# ============================================================================

# ---- Bagimlilik ----
import subprocess, sys, os, re, csv, time
try:
    import medmnist
except ImportError:
    print("medmnist kuruluyor..."); subprocess.run([sys.executable,"-m","pip","install","-q","medmnist"])
    import medmnist
from medmnist import INFO
from google.colab import drive
drive.mount('/content/drive')

REPO = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/Inference_pytorch"
OUT  = "/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures/SAA_crossdataset.csv"
os.chdir(REPO)
print("Calisma dizini:", os.getcwd())

DATASETS = ["pneumoniamnist", "dermamnist", "bloodmnist", "retinamnist"]
COMMON = ["--model","VGG8","--mode","WAGE","--inference","1",
          "--subArray","128","--parallelRead","128","--seed","117",
          "--skip_hw_estimate","1"]

# Pneumonia'da turetilen SAA tahsisleri (yeniden tune EDILMEDEN uygulanir)
allocs = {
    "ALL12"    : "12,12,12,12,12,12,12,12",
    "ALL5"     : "5,5,5,5,5,5,5,5",
    "MIX_C"    : "5,12,12,12,12,12,12,5",   # SAA avg 10.25
    "MIX_E"    : "5,8,12,12,12,12,12,5",     # SAA avg 9.75
    "SAA_B11.0": "8,12,12,12,12,12,12,8",    # SAA avg 11.0 (yeni kesif)
}

def parse_metrics(text):
    acc=None
    m=re.findall(r'Accuracy:\s*([0-9]+)\s*/\s*([0-9]+)', text)
    if m: c,t=m[-1]; acc=round(100.0*int(c)/int(t),2)
    bacc=None
    m=re.findall(r'Balanced\s*Accuracy:\s*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
    if m: bacc=float(m[-1])
    auc=None
    m=re.findall(r'\bAUC:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if m: auc=float(m[-1])
    return acc,bacc,auc

# devam destegi: zaten kayitli (dataset,policy) ciftlerini atla
done=set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for row in csv.DictReader(f):
            if row.get("auc") not in (None,"","None"):
                done.add((row["dataset"],row["policy"]))
    print("Zaten tamamlanmis:", len(done), "kosum (atlanacak)")

def run_one(ds, name, spec, timeout=5400):
    cmd=["python","inference.py","--dataset",ds,"--adc_per_layer",spec]+COMMON
    print(f"\n{'='*60}\n[{ds}] {name}  ADC={spec}")
    t0=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        log=r.stdout+"\n"+r.stderr
    except Exception as e:
        log=f"[HATA] {e}"
    dt=time.time()-t0
    acc,bacc,auc=parse_metrics(log)
    avg=sum(int(x) for x in spec.split(","))/len(spec.split(","))
    for line in log.splitlines():
        if any(k in line for k in ["Accuracy:","Balanced","AUC:"]): print("   ",line.strip())
    print(f"   --> acc={acc} bacc={bacc} auc={auc} avg_bits={avg:.3f} ({dt:.0f}s)")
    return {"dataset":ds,"policy":name,"adc_per_layer":spec,"avg_bits":round(avg,3),
            "accuracy_pct":acc,"balanced_accuracy_pct":bacc,"auc":auc}

# mevcut satirlari yukle (devam icin)
rows=[]
if os.path.exists(OUT):
    with open(OUT) as f: rows=list(csv.DictReader(f))

def save():
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["dataset","policy","adc_per_layer","avg_bits",
                         "accuracy_pct","balanced_accuracy_pct","auc"])
        w.writeheader()
        for r_ in rows: w.writerow(r_)

for ds in DATASETS:
    for name,spec in allocs.items():
        if (ds,name) in done:
            print(f"[ATLA] {ds}/{name} zaten var"); continue
        rows=[r for r in rows if not (r["dataset"]==ds and r["policy"]==name)]
        rows.append(run_one(ds,name,spec))
        save()   # her kosumdan sonra kaydet

print("\n"+"="*60+"\nTAMAMLANDI. Kaydedildi:",OUT)
import pandas as pd
df=pd.DataFrame(rows)
for ds in DATASETS:
    sub=df[df.dataset==ds].copy()
    if len(sub):
        sub["avg_bits"]=sub["avg_bits"].astype(float)
        print(f"\n--- {ds} ---")
        print(sub.sort_values("avg_bits")[["policy","avg_bits","accuracy_pct","balanced_accuracy_pct","auc"]].to_string(index=False))
