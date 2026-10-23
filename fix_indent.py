with open('scripts/11_train_samtseg.py', 'rb') as f:
    content = f.read()

replacements = {
    b'        "run_name": args.run_name,': b'    "run_name": args.run_name,',
    b'        "params_M": sum(p.numel() for p in get_model(num_classes=10, backbone=args.encoder, pretrained=False).parameters()) / 1e6,': b'    "params_M": sum(p.numel() for p in get_model(num_classes=10, backbone=args.encoder, pretrained=False).parameters()) / 1e6,',
    b'        "best_val_miou": best_miou,': b'    "best_val_miou": best_miou,',
    b'        "best_epoch": best_epoch,': b'    "best_epoch": best_epoch,',
    b'        "config": {"encoder": args.encoder, "pretrained": not args.no_pretrained': b'    "config": {"encoder": args.encoder, "pretrained": not args.no_pretrained',
    b'                    "epochs": args.epochs, "batch_size": args.batch_size},': b'        "epochs": args.epochs, "batch_size": args.batch_size},',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('scripts/11_train_samtseg.py', 'wb') as f:
    f.write(content)

print('Fixed')