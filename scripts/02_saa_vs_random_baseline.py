# ============================================================================
#  MADDE 2 — SAA vs RANDOM vs UNIFORM baseline (VGG8, PneumoniaMNIST)
#  --------------------------------------------------------------------------
#  Hakemin MERKEZI itirazi: "SAA sadece sort+greedy fill; matched ortalama-bit
#  butcesinde RANDOM tahsisten iyi mi?" Bunu kapatmak 'yontem' iddiasini mesru
#  kilar.
#
#  Iki butce: avg~10.25 (MIX_C'nin butcesi) ve avg~9.75 (MIX_E'nin butcesi).
#  Her butcede:
#    - SAA tahsisi (duyarlilik-gudumlu; MIX_C / MIX_E)
#    - 5 RANDOM tahsis (ayni ortalama bite yakin; katmanlara rastgele dagitim)
#    - 1 UNIFORM-yuvarlama (tum katmanlar = round(avg))
#  SAA, random'larin dagilimini (mean±std) ANLAMLI geciyor mu?
#
#  8 katman, bit secenekleri {5,8,12}. Random tahsis ortalama biti hedefe
#  (+-0.4 bit) yaklasana kadar ornekle. seed=117 sabit (inference det.).
#
#  Kosum sayisi: 2 butce x (5 random + 1 uniform) = 12 yeni kosum (+ SAA zaten
#  var, tekrar kosariz tutarlilik icin = 14). VGG8 ~610s -> ~2.4 saat. Ara-kayit.
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

def random_alloc(target_avg, rng, tol=0.4):
    # ortalama biti target_avg'e yakin rastgele {5,8,12} vektoru uret
    for _ in range(2000):
        b=[rng.choice(BITS) for _ in range(NLAYERS)]
        if abs(sum(b)/NLAYERS - target_avg) <= tol:
            return b
    return b

# butceler: (isim, SAA_spec, target_avg)
budgets=[
 ("B10.25", "5,12,12,12,12,12,12,5", 10.25),   # MIX_C
 ("B9.75",  "5,8,12,12,12,12,12,5",  9.75),     # MIX_E
]

CKPT=os.path.join(OUTDIR,"saa_baseline_ckpt.json")
rows=json.load(open(CKPT)) if os.path.exists(CKPT) else []
done={r["tag"] for r in rows if r.get("auc") is not None}
def save(): json.dump(rows,open(CKPT,"w"),indent=1)
def do(spec,tag):
    if tag in done: print(f"[ATLA] {tag}"); return
    rows.append(run(spec,tag)); save()

for bname, saa_spec, tavg in budgets:
    # SAA
    do(saa_spec, f"{bname}_SAA")
    # uniform-yuvarlama
    ub=round(tavg); ub=min(BITS,key=lambda x:abs(x-tavg))  # en yakin izinli bit
    do(vec([ub]*NLAYERS), f"{bname}_UNIFORM{ub}")
    # 5 random (ayni ortalamaya yakin), sabit rng tohumu -> tekrarlanabilir
    rng=random.Random(2026 + int(tavg*100))
    for k in range(5):
        b=random_alloc(tavg, rng)
        do(vec(b), f"{bname}_RAND{k}")

# ---------- ANALIZ ----------
import statistics as st
print("\n"+"="*60+"\nSONUC — SAA vs RANDOM vs UNIFORM")
for bname, saa_spec, tavg in budgets:
    print(f"\n--- Butce {bname} (hedef avg~{tavg}) ---")
    saa=next((r for r in rows if r["tag"]==f"{bname}_SAA"),None)
    uni=next((r for r in rows if r["tag"].startswith(f"{bname}_UNIFORM")),None)
    rands=[r for r in rows if r["tag"].startswith(f"{bname}_RAND") and r.get("auc") is not None]
    if saa: print(f"  SAA       avg={saa['avg_bits']:.2f}  acc={saa['acc']}  auc={saa['auc']}")
    if uni: print(f"  UNIFORM   avg={uni['avg_bits']:.2f}  acc={uni['acc']}  auc={uni['auc']}")
    if rands:
        racc=[r['acc'] for r in rands if r['acc'] is not None]
        rauc=[r['auc'] for r in rands if r['auc'] is not None]
        print(f"  RANDOM(n={len(rands)}) acc={st.mean(racc):.2f}±{st.pstdev(racc):.2f} (min {min(racc)}, max {max(racc)})")
        print(f"              auc={st.mean(rauc):.4f}±{st.pstdev(rauc):.4f} (min {min(rauc)}, max {max(rauc)})")
        if saa and saa['auc'] is not None:
            better=sum(1 for a in rauc if saa['auc']>a)
            print(f"  -> SAA AUC, {len(rauc)} random'in {better} tanesinden YUKSEK")
            z=(saa['auc']-st.mean(rauc))/ (st.pstdev(rauc) if st.pstdev(rauc)>0 else 1)
            print(f"  -> SAA, random ortalamasindan {z:+.2f} std uzakta")

csvp=os.path.join(OUTDIR,"saa_baseline_results.csv")
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["tag","spec","avg_bits","acc","bacc","auc"]); w.writeheader()
    for r in rows: w.writerow(r)
print("\nCSV:",csvp,"\nTAMAMLANDI.")
