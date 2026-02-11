import os
import os.path as osp
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main(run_dir: str):
    
    results_dir = osp.join(run_dir, "results")
    csv_path = osp.join(results_dir, "history.csv")

    if not osp.exists(csv_path):
        raise FileNotFoundError(
            f"Non trovo {csv_path}. Assicurati di aver finito almeno 1 training e che history.csv sia stato scritto."
        )

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    out1 = osp.join(results_dir, "loss_curve.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    out2 = osp.join(results_dir, "acc_curve.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    print("[OK] Salvati:")
    print(" -", out1)
    print(" -", out2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./runs/tox21/test",
                        help="Path della run (es: ./runs/tox21/test)")
    args = parser.parse_args()
    main(args.run_dir)
