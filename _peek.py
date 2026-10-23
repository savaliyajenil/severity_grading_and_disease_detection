import csv, sys
names = sys.argv[1:]
print(f"{'run':<16}{'ep':>5}{'mIoU':>9}{'disIoU':>9}{'sevMAE':>9}")
print("-" * 48)
for n in names:
    rows = list(csv.DictReader(open(f"results/{n}/training_log.csv")))
    b = max(rows, key=lambda r: float(r["Val_mIoU"]))
    print(f"{n:<16}{b['Epoch']:>5}{float(b['Val_mIoU']):>9.4f}"
          f"{float(b['Val_Disease_mIoU']):>9.4f}{float(b['Sev_MAE']):>9.4f}")
