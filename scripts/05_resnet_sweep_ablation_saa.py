# ============================================================================
#  ResNet18 — ADC SWEEP + LEAVE-ONE-LOW ABLASYON + SAA  (cok-mimari kaniti)
#  --------------------------------------------------------------------------
#  Pneumonia'da egitilmis ResNet18 (log/ResNet18_pneumoniamnist.pth) uzerinde:
#   A) Uniform sweep: ALL5, ALL8, ALL12         -> 5-bit cokusu ResNet'te de var mi?
#   B) Leave-one-low ablasyon: 21 katmanin her biri tek tek 5-bit (gerisi 12)
#      -> her katmanin AUC duyarliligi (robust/fragile partisyon)
#   C) SAA: ablasyondan turet. Katmanlari AUC-dususune gore sirala; en robust
#      olanlari 5-bit'e indir, hedef ortalama bit'e ulas. Birkac hedef -> Pareto.
#      Uniform ALL12 ile karsilastir (ayni AUC, daha az bit?).
#
#  21 katman: conv1, layer1-4 (her biri 2 blok x conv1/conv2 + bazilarinda
#  downsample.0), fc. ADC katman sirasi list_adc_layers() ile AYNI.
#
#  Runtime: T4. Kosum ~16-20 sn (ResNet hizli). Toplam:
#    sweep 3 + ablasyon 21 + SAA ~6 = ~30 kosum x ~18 sn ≈ 9-10 dk. ARA-KAYIT var.
# ============================================================================
import subprocess, sys, os, re, csv, time, json
try:
    import medmnist
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","medmnist"]); import medmnist
from google.colab import drive
drive.mount('/content/drive')

REPO="/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/Inference_pytorch"
OUTDIR="/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures"
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(REPO)

DATASET="pneumoniamnist"
NLAYERS=21
BASE=["--model","ResNet18","--mode","WAGE","--inference","1",
      "--subArray","128","--parallelRead","128","--seed","117","--skip_hw_estimate","1"]

def parse(text):
    acc=None; m=re.findall(r'Accuracy:\s*([0-9]+)\s*/\s*([0-9]+)',text)
    if m: c,t=m[-1]; acc=round(100*int(c)/int(t),2)
    bacc=None; m=re.findall(r'Balanced\s*Accuracy:\s*([0-9]+\.?[0-9]*)',text,re.I)
    if m: bacc=float(m[-1])
    auc=None; m=re.findall(r'\bAUC:\s*([0-9]*\.?[0-9]+)',text,re.I)
    if m: auc=float(m[-1])
    return acc,bacc,auc

def run(spec, tag, timeout=1800):
    cmd=["python","inference.py","--dataset",DATASET,"--adc_per_layer",spec]+BASE
    t0=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout); log=r.stdout+"\n"+r.stderr
    except Exception as e: log=f"[HATA] {e}"
    acc,bacc,auc=parse(log); dt=time.time()-t0
    print(f"  {tag:22} avg={sum(int(x) for x in spec.split(','))/NLAYERS:5.2f}  acc={acc}  bacc={bacc}  auc={auc}  ({dt:.0f}s)")
    if auc is None:  # hata ayikla
        for ln in log.splitlines():
            if any(k in ln for k in ["Error","Traceback","assert","seek","Module"]): print("      !",ln.strip())
    return {"tag":tag,"adc_per_layer":spec,"avg_bits":round(sum(int(x) for x in spec.split(","))/NLAYERS,3),
            "acc":acc,"bacc":bacc,"auc":auc}

def vec(bits): return ",".join(str(b) for b in bits)

# ara-kayit
CKPT=os.path.join(OUTDIR,"resnet_saa_ckpt.json")
results=[]
if os.path.exists(CKPT):
    results=json.load(open(CKPT)); print(f"Ara-kayit yuklendi: {len(results)} kosum")
done={r["tag"] for r in results if r.get("auc") is not None}
def save():
    json.dump(results,open(CKPT,"w"),indent=1)

def do(spec,tag):
    if tag in done: print(f"  [ATLA] {tag}"); 
    else:
        results.append(run(spec,tag)); save()

# ---------- A) UNIFORM SWEEP ----------
print("\n=== A) UNIFORM SWEEP (ResNet18) ===")
do(vec([5]*NLAYERS),  "ALL5")
do(vec([8]*NLAYERS),  "ALL8")
do(vec([12]*NLAYERS), "ALL12")

all12 = next((r for r in results if r["tag"]=="ALL12"), None)
auc12 = all12["auc"] if all12 else None
print(f"  (ALL12 referans AUC = {auc12})")

# ---------- B) LEAVE-ONE-LOW ABLASYON ----------
# her katman i: o katman 5-bit, gerisi 12-bit. AUC ne kadar duserse katman o kadar FRAGILE.
print("\n=== B) LEAVE-ONE-LOW ABLASYON (21 katman) ===")
for i in range(NLAYERS):
    bits=[12]*NLAYERS; bits[i]=5
    do(vec(bits), f"L1L_{i:02d}")

# duyarlilik tablosu: AUC dususu (auc12 - auc_i). Buyuk dusus = fragile.
abl=[]
for i in range(NLAYERS):
    r=next((x for x in results if x["tag"]==f"L1L_{i:02d}"),None)
    if r and r["auc"] is not None and auc12 is not None:
        abl.append((i, r["auc"], round(auc12-r["auc"],4)))
abl_sorted=sorted(abl, key=lambda z: z[2])  # kucuk dusus (robust) -> once
print("\n  --- Duyarlilik (AUC dususune gore, ROBUST -> FRAGILE) ---")
print("  layer_idx  auc_when_low  auc_drop")
for i,a,d in abl_sorted:
    print(f"    {i:02d}        {a:.4f}       {d:+.4f}")

# en robust katmanlar (dusuk auc_drop) 5-bit'e indirilmeye aday.
robust_order=[i for i,a,d in abl_sorted]  # robust -> fragile

# ---------- C) SAA TAHSISLERI ----------
# Hedef ortalama bit icin: en robust K katmani 5-bit yap, gerisi 12-bit.
# avg = (K*5 + (N-K)*12)/N. Birkac K deneyip Pareto cikar.
print("\n=== C) SAA TAHSISLERI (ablasyondan turetildi) ===")
def saa_alloc(K):
    bits=[12]*NLAYERS
    for i in robust_order[:K]: bits[i]=5
    return bits
for K in [4,6,8,10,12]:
    bits=saa_alloc(K)
    avg=sum(bits)/NLAYERS
    do(vec(bits), f"SAA_K{K:02d}_avg{avg:.1f}")

# ---------- OZET + CSV ----------
print("\n"+"="*60+"\nOZET (ResNet18)")
import pandas as pd
df=pd.DataFrame(results)
if len(df):
    df=df[["tag","avg_bits","acc","bacc","auc"]].copy()
    print(df.to_string(index=False))
    csvp=os.path.join(OUTDIR,"resnet_saa_results.csv")
    df.to_csv(csvp,index=False)
    print("\nCSV:",csvp)
    # robust sirasini da kaydet
    json.dump({"auc12":auc12,"robust_order":robust_order,
               "ablation":[{"layer":i,"auc":a,"drop":d} for i,a,d in abl_sorted]},
              open(os.path.join(OUTDIR,"resnet_ablation.json"),"w"),indent=1)
    print("Ablasyon:",os.path.join(OUTDIR,"resnet_ablation.json"))
print("\nTAMAMLANDI.")
