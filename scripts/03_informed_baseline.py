# ============================================================================
#  MADDE 2 (ek) — BILGILENDIRILMIS baseline: SAA vs KORUMALI-random vs naif-kural
#  --------------------------------------------------------------------------
#  Hakem (2. tur): "random baseline hasir-adam; classifier.0'i (felaket katman,
#  LOW6 AUC 0.363) 5-bit'e koyabildigi icin cokuyor. SAA'nin INCE SIRALAMASI
#  random'dan iyi mi, yoksa sadece bir felaket hatasindan mi kaciniyor?"
#
#  Bu kosum iki ADIL baseline ekler:
#   (1) KORUMALI-random: classifier.0 (idx6) HER ZAMAN 12-bit; geri kalan
#       katmanlara random {5,8,12}, hedef ortalama bite yakin.
#   (2) NAIF-kural: "sadece iki uc katmani (features.0, classifier.2) dusur"
#       = MIX_C'nin ta kendisi ASLINDA; o yuzden naif-kural olarak "en robust
#       3 katmani dusur" varyantini da ekliyoruz (idx0,idx7,idx4).
#
#  Eger SAA (MIX_C/MIX_E), KORUMALI-random'i da geciyorsa -> ince siralama
#  gercekten degerli. Gecmiyorsa -> dürüstce "ana kazanc felaket-katman
#  korumasi" deriz. Her iki sonuc da hakemi yanitlar.
#
#  2 butce x 5 korumali-random + 2 naif-kural = 12 kosum. VGG8 ~610s -> ~2 saat.
#  Ara-kayit var. seed=117 sabit (inference det.).
# ============================================================================
import subprocess, sys, os, re, csv, time, json, random
try:
    import medmnist
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","medmnist"]); import medmnist
from google.colab import drive
drive.mount('/content/drive')

REPO="/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/Inference_pytorch"
OUTDIR="/content/drive/MyDrive/DNN_Neurosimv14_makale/DNN_NeuroSim_V1.4/paper_figures"
os.makedirs(OUTDIR,exist_ok=True); os.chdir(REPO)

NLAYERS=8
BITS=[5,8,12]
PROTECT_IDX=6   # classifier.0 - felaket katman, her zaman 12-bit
BASE=["--dataset","pneumoniamnist","--model","VGG8","--mode","WAGE","--inference","1",
      "--subArray","128","--parallelRead","128","--seed","117","--skip_hw_estimate","1"]

def parse(text):
    acc=None; m=re.findall(r'Accuracy:\s*([0-9]+)\s*/\s*([0-9]+)',text)
    if m: c,t=m[-1]; acc=round(100*int(c)/int(t),2)
    bacc=None; m=re.findall(r'Balanced\s*Accuracy:\s*([0-9]+\.?[0-9]*)',text,re.I)
    if m: bacc=float(m[-1])
    auc=None; m=re.findall(r'\bAUC:\s*([0-9]*\.?[0-9]+)',text,re.I)
    if m: auc=float(m[-1])
    return acc,bacc,auc

def run(spec, tag):
    cmd=["python","inference.py","--adc_per_layer",spec]+BASE
    print(f"\n{tag}  [{spec}]"); t0=time.time()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=5400); log=r.stdout+"\n"+r.stderr
    except Exception as e: log=f"[HATA] {e}"
    acc,bacc,auc=parse(log); dt=time.time()-t0
    avg=sum(int(x) for x in spec.split(","))/NLAYERS
    print(f"   avg={avg:.3f} acc={acc} bacc={bacc} auc={auc} ({dt:.0f}s)")
    return {"tag":tag,"spec":spec,"avg_bits":round(avg,3),"acc":acc,"bacc":bacc,"auc":auc}

def vec(b): return ",".join(str(x) for x in b)

def protected_random(target_avg, rng, tol=0.4):
    # classifier.0 (idx6) HER ZAMAN 12; geri kalana random, ortalama hedefe yakin
    for _ in range(4000):
        b=[rng.choice(BITS) for _ in range(NLAYERS)]
        b[PROTECT_IDX]=12
        if abs(sum(b)/NLAYERS - target_avg) <= tol:
            return b
    return b

CKPT=os.path.join(OUTDIR,"informed_baseline_ckpt.json")
rows=json.load(open(CKPT)) if os.path.exists(CKPT) else []
done={r["tag"] for r in rows if r.get("auc") is not None}
def save(): json.dump(rows,open(CKPT,"w"),indent=1)
def do(spec,tag):
    if tag in done: print(f"[ATLA] {tag}"); return
    rows.append(run(spec,tag)); save()

budgets=[("B10.25",10.25),("B9.75",9.75)]

for bname,tavg in budgets:
    rng=random.Random(7000+int(tavg*100))
    for k in range(5):
        b=protected_random(tavg,rng)
        do(vec(b), f"{bname}_PROTRAND{k}")

# NAIF-kural: en robust-K katmani dusur (ablasyon sirasina gore idx0,idx7 sonra idx4)
# K=2 -> MIX_C ile ayni ([5 end,12...]) zaten var; K=3 naif: idx0,idx7,idx4 -> 5
naif3=[12]*NLAYERS
for i in [0,7,4]: naif3[i]=5   # en robust 3 (idx4 features.10 orta-robust)
do(vec(naif3), "NAIF_drop3_robust")
# naif 'sadece uc 2 katman' zaten MIX_C; referans icin tekrar
do(vec([5,12,12,12,12,12,12,5]), "NAIF_drop2_ends(=MIX_C)")

# ---------- ANALIZ ----------
import statistics as st
# mevcut SAA/random sonuclarini da oku (onceki baseline kosumundan)
prev=os.path.join(OUTDIR,"saa_baseline_results.csv")
saa={}
if os.path.exists(prev):
    with open(prev) as f:
        for r in csv.DictReader(f):
            if "_SAA" in r["tag"] and r.get("auc"): saa[r["tag"].split("_")[0]]=float(r["auc"])

print("\n"+"="*60+"\nSONUC — SAA vs KORUMALI-random vs naif-kural")
for bname,tavg in budgets:
    print(f"\n--- Butce {bname} (avg~{tavg}) ---")
    if bname in saa: print(f"  SAA               auc={saa[bname]:.4f}")
    pr=[r for r in rows if r["tag"].startswith(f"{bname}_PROTRAND") and r.get("auc")]
    if pr:
        aucs=[r["auc"] for r in pr]
        print(f"  KORUMALI-random(n={len(pr)})  auc={st.mean(aucs):.4f}±{st.pstdev(aucs):.4f} (min {min(aucs)}, max {max(aucs)})")
        if bname in saa:
            better=sum(1 for a in aucs if saa[bname]>a)
            print(f"    -> SAA, {len(aucs)} korumali-random'in {better} tanesinden YUKSEK")
naif=[r for r in rows if r["tag"].startswith("NAIF") and r.get("auc")]
for r in naif: print(f"  {r['tag']:26} avg={r['avg_bits']} auc={r['auc']}")

csvp=os.path.join(OUTDIR,"informed_baseline_results.csv")
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["tag","spec","avg_bits","acc","bacc","auc"]); w.writeheader()
    for r in rows: w.writerow(r)
print("\nCSV:",csvp,"\nTAMAMLANDI.")
