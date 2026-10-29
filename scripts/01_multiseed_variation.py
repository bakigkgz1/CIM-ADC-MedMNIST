# ============================================================================
#  MADDE 1 (v2) — SEED VARYANSI vari>0 ile — VGG8, PneumoniaMNIST
#  --------------------------------------------------------------------------
#  Onceki kosum std=0 verdi cunku inference deterministik (agirlik sabit,
#  vari=0). GERCEK stokastik kaynak konduktans varyasyonudur. Bu kosum her
#  config'i vari=0.1'de 5 FARKLI seed ile kosar -> gercek mean+-std ve
#  'donanim gurultusu altinda' paired non-inferiority.
#
#  Bu, bir RELIABILITY makalesi icin dogru test: cip-cip varyasyonu altinda
#  layer-wise ADC tahsisi hala ALL12 ile ayirt edilemez mi?
#
#  6 config x 5 seed = 30 kosum. VGG8 Pneumonia ~610s -> ~5 saat. Ara-kayit var.
#  Seedler: 117,17,42,1234,2025 | vari=0.1
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
os.makedirs(OUTDIR,exist_ok=True); os.chdir(REPO)

VARI="0.1"
SEEDS=[117,17,42,1234,2025]
CONFIGS={
 "ALL5":    "5,5,5,5,5,5,5,5",
 "ALL8":    "8,8,8,8,8,8,8,8",
 "ALL12":   "12,12,12,12,12,12,12,12",
 "MIX_E":   "5,8,12,12,12,12,12,5",
 "MIX_C":   "5,12,12,12,12,12,12,5",
 "SAA_B11": "8,12,12,12,12,12,12,8",
}
BASE=["--dataset","pneumoniamnist","--model","VGG8","--mode","WAGE","--inference","1",
      "--subArray","128","--parallelRead","128","--skip_hw_estimate","1","--vari",VARI]

def parse(text):
    acc=None; m=re.findall(r'Accuracy:\s*([0-9]+)\s*/\s*([0-9]+)',text)
    if m: c,t=m[-1]; acc=round(100*int(c)/int(t),2)
    bacc=None; m=re.findall(r'Balanced\s*Accuracy:\s*([0-9]+\.?[0-9]*)',text,re.I)
    if m: bacc=float(m[-1])
    auc=None; m=re.findall(r'\bAUC:\s*([0-9]*\.?[0-9]+)',text,re.I)
    if m: auc=float(m[-1])
    return acc,bacc,auc

CKPT=os.path.join(OUTDIR,"multiseed_vari_ckpt.json")
rows=json.load(open(CKPT)) if os.path.exists(CKPT) else []
done={(r["config"],r["seed"]) for r in rows if r.get("auc") is not None}
if done: print("Ara-kayit:",len(done),"kosum tamam (atlanacak)")
def save(): json.dump(rows,open(CKPT,"w"),indent=1)

total=len(CONFIGS)*len(SEEDS); i=0
for cfg,spec in CONFIGS.items():
    for sd in SEEDS:
        i+=1
        if (cfg,sd) in done: print(f"[{i}/{total}] [ATLA] {cfg} seed={sd}"); continue
        cmd=["python","inference.py","--adc_per_layer",spec,"--seed",str(sd)]+BASE
        print(f"\n[{i}/{total}] {cfg} seed={sd} vari={VARI}"); t0=time.time()
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=5400); log=r.stdout+"\n"+r.stderr
        except Exception as e: log=f"[HATA] {e}"
        acc,bacc,auc=parse(log); dt=time.time()-t0
        print(f"   acc={acc} bacc={bacc} auc={auc} ({dt:.0f}s)")
        rows=[x for x in rows if not (x["config"]==cfg and x["seed"]==sd)]
        rows.append({"config":cfg,"seed":sd,"vari":VARI,"acc":acc,"bacc":bacc,"auc":auc}); save()

import statistics as st
print("\n"+"="*60+f"\nSONUC (vari={VARI}, 5 seed) — mean +- std")
def grp(cfg,key): return [r[key] for r in rows if r["config"]==cfg and r.get(key) is not None]
order=["ALL5","ALL8","MIX_E","MIX_C","SAA_B11","ALL12"]
for cfg in order:
    line=f"  {cfg:8}"
    for key in ["acc","bacc","auc"]:
        v=grp(cfg,key)
        if len(v)>=2: line+=f"  {key}={st.mean(v):.4f}±{st.pstdev(v):.4f}"
    print(line)

print("\n--- PAIRED fark vs ALL12 (ayni seed, vari={}) ---".format(VARI))
for cand in ["MIX_C","SAA_B11"]:
    print(f"  [{cand} - ALL12]")
    for key in ["acc","auc"]:
        diffs=[]
        for sd in SEEDS:
            a=next((r[key] for r in rows if r["config"]=="ALL12" and r["seed"]==sd and r.get(key) is not None),None)
            m=next((r[key] for r in rows if r["config"]==cand and r["seed"]==sd and r.get(key) is not None),None)
            if a is not None and m is not None: diffs.append(m-a)
        if len(diffs)>=2:
            md=st.mean(diffs); sdv=st.pstdev(diffs); sem=sdv/(len(diffs)**0.5)
            print(f"     {key}: mean_diff={md:+.4f}  std={sdv:.4f}  95%CI~[{md-2*sem:+.4f},{md+2*sem:+.4f}]  diffs={[round(d,4) for d in diffs]}")

csvp=os.path.join(OUTDIR,"multiseed_vari_results.csv")
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["config","seed","vari","acc","bacc","auc"]); w.writeheader()
    for r in rows: w.writerow(r)
print("\nCSV:",csvp,"\nTAMAMLANDI.")
