# CIM-ADC-MedMNIST

Code and result data for the manuscript:

**"Layer-wise Mixed-Precision ADC Allocation for Reliable Compute-in-Memory Inference: A Task-Dependent Design-Space Analysis"**
Baki Gokgoz. ORCID: 0000-0001-8166-6282.

Analog compute-in-memory (CIM) accelerators validate ADC bit-width budgets on natural-image benchmarks such as CIFAR-10. This work shows those budgets do **not** transfer to medical imaging: a uniform 5-bit ADC that keeps 91.5% CIFAR-10 accuracy drives four MedMNIST tasks toward near-chance AUC. A leave-one-low ablation reveals a robust/fragile layer partition, which a **sensitivity-driven automatic allocation (SAA)** turns into a per-layer ADC budget that recovers 12-bit reliability at fewer average bits, generalizes across tasks and architectures (VGG8 and ResNet18), beats matched-budget random and catastrophic-layer-protected random allocations, and is non-inferior to uniform 12-bit under conductance variation across five seeds.

All experiments use DNN+NeuroSim V1.4 (WAGE mode, VGG8 / ResNet18, subArray=128, parallelRead=128).

## Data (`data/`)

| File | Manuscript | Contents |
|------|-----------|----------|
| `adc_sweep_4datasets.csv` | Table 2 | Uniform 5/8/12-bit ADC sweep, 4 MedMNIST tasks |
| `variation_sweep_pneumonia.csv` | Table 3 | ADC x conductance variation, PneumoniaMNIST |
| `saa_crossdataset_transfer.csv` | Table 6 | Pneumonia-tuned allocation applied to 4 tasks |
| `multiseed_variation_5seed.csv` | Table 7 | 6 configs x 5 seeds at vari=0.1 (non-inferiority) |
| `saa_vs_random_baseline.csv` | Table 8 | SAA vs matched-budget random/uniform |
| `informed_baseline_results.csv` | Table 8 | SAA vs catastrophic-layer-protected random + naive rule |
| `resnet18_sweep_ablation_saa.csv` | Table 9 | ResNet18 sweep + 21-layer ablation + SAA |
| `resnet18_ablation.json` | - | ResNet18 per-layer sensitivity detail |

## Scripts (`scripts/`)

Google Colab cells (run on a NeuroSim V1.4 tree with the per-layer ADC helper `utee/layer_adc.py`):

| Script | Produces |
|--------|----------|
| `01_multiseed_variation.py` | Table 7 (multi-seed robustness, vari=0.1) |
| `02_saa_vs_random_baseline.py` | Table 8 (SAA vs random/uniform) |
| `03_informed_baseline.py` | Table 8 (SAA vs protected-random + naive rule) |
| `04_crossdataset_transfer.py` | Table 6 (cross-dataset transfer) |
| `05_resnet_sweep_ablation_saa.py` | Table 9 (ResNet18 cross-architecture) |
| `06_adc_sweep_tables.py` | Tables 2-3 (uniform sweep + variation) |
| `07_saa_pareto.py` | Table 5 (SAA Pareto front) |

## SAA in brief

Per-layer ADC sensitivity is s_i = AUC(ALL12) - AUC(LOW_i), where LOW_i lowers only layer i to 5-bit. SAA starts all layers at 12-bit and, given a target average bit-width, greedily lowers the most robust layers (smallest s_i) one level at a time (12->8->5) until the target is met. This reproduces the hand-selected MIX_C and MIX_E allocations from a single scalar parameter.

## Reproducing

1. Clone a DNN+NeuroSim V1.4 tree; add `utee/layer_adc.py` (functions `parse_adc_per_layer`, `apply_adc_per_layer`, `list_adc_layers`); inference is invoked with `--adc_per_layer "b1,b2,..."`.
2. Train WAGE weights per dataset into `./log/VGG8_<dataset>.pth` (and `./log/ResNet18_<dataset>.pth`).
3. Run the scripts in `scripts/` (Colab, T4 GPU). Each writes its CSV into `paper_figures/`.

## Citation

If you use this code or data, please cite the manuscript above.

## License

Released for academic review and reuse. Data files are simulation outputs.
