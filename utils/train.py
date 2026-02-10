import os
import torch
import torch.nn.functional as F
import os.path as osp
import json
import matplotlib.pyplot as plt
import numpy as np
import csv

from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy
from torch.utils.tensorboard import SummaryWriter



def confusion_matrix_torch(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    y_true, y_pred: Tensor 1D su CPU (dtype long)
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t, p] += 1
    return cm.numpy()


def save_confusion_matrix(cm: np.ndarray, out_png: str, class_names=None, normalize: bool = False):
    os.makedirs(osp.dirname(out_png), exist_ok=True)

    if normalize:
        plot_cm = cm.astype(np.float64)
        row_sum = plot_cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        plot_cm = plot_cm / row_sum
        fmt = ".2f"
    else:
        plot_cm = cm.astype(np.int64)   # <-- QUI: interi veri
        fmt = "d"

    plt.figure()
    plt.imshow(plot_cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(cm.shape[0])
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = plot_cm.max() * 0.6 if plot_cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(plot_cm[i, j], fmt),
                ha="center", va="center",
                color="white" if plot_cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def train_epoch_classifier(model, train_loader, len_train, optimizer, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        #optimizer.zerograd()
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train

@torch.no_grad()
def test_classifier(model, loader, device, return_preds: bool = False):
    model.eval()

    y = torch.tensor([], dtype=torch.long, device=device)
    yp = torch.tensor([], dtype=torch.long, device=device)

    loss_all = 0.0
    for data in loader:
        data = data.to(device)
        logits, _ = model(data.x, data.edge_index, batch=data.batch)

        loss = F.nll_loss(F.log_softmax(logits, dim=-1), data.y)
        pred = logits.max(dim=1)[1]

        y = torch.cat([y, data.y])
        yp = torch.cat([yp, pred])

        loss_all += data.num_graphs * loss.item()

    acc = accuracy(y, yp).item() if hasattr(accuracy(y, yp), "item") else float(accuracy(y, yp))
    prec = precision(y, yp, model.num_output).mean().item()
    rec  = recall(y, yp, model.num_output).mean().item()
    f1   = f1_score(y, yp, model.num_output).mean().item()

    out = (acc, prec, rec, f1, loss_all)

    if return_preds:
        return (*out, y.detach().cpu(), yp.detach().cpu())
    return out




def train_cycle_classifier(
    task,
    train_loader,
    val_loader,
    test_loader,
    len_train,
    len_val,
    len_test,
    model,
    optimizer,
    device,
    base_path,
    epochs,
):
    best_acc = (0.0, 0.0)
    writer = SummaryWriter(osp.join(base_path, "plots"))

    ckpt_dir = osp.join(base_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # dove salvo tabella e confusion matrix
    results_dir = osp.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    history = []
    history_jsonl = osp.join(results_dir, "history.jsonl")

    for epoch in range(epochs):
        loss = train_epoch_classifier(model, train_loader, len_train, optimizer, device)
        writer.add_scalar("Loss/train", loss, epoch)

        train_acc, train_prec, train_rec, train_f1, train_loss_sum = test_classifier(
            model, train_loader, device, return_preds=False
        )
        val_acc, val_prec, val_rec, val_f1, val_loss_sum, y_val, yp_val = test_classifier(
            model, val_loader, device, return_preds=True
        )

        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Loss/val", val_loss_sum / len_val, epoch)

        row = {
            "epoch": epoch,
            "train_loss": float(loss),
            "train_acc": float(train_acc),
            "train_prec": float(train_prec),
            "train_rec": float(train_rec),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss_sum / len_val),
            "val_acc": float(val_acc),
            "val_prec": float(val_prec),
            "val_rec": float(val_rec),
            "val_f1": float(val_f1),
        }
        history.append(row)

        # append JSONL (così non perdi nulla se si interrompe)
        with open(history_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"Epoch: {epoch}, Loss: {loss:.5f}")
        print(f"Train -> Acc: {train_acc:.5f}  Rec: {train_rec:.5f}  Prec: {train_prec:.5f}  F1: {train_f1:.5f}")
        print(f"Val   -> Acc: {val_acc:.5f}  Rec: {val_rec:.5f}  Prec: {val_prec:.5f}  F1: {val_f1:.5f}")

        # salva best model (su val acc) + confusion matrix
        if best_acc[1] < val_acc:
            best_acc = (train_acc, val_acc)

            ckpt_path = osp.join(ckpt_dir, model.__class__.__name__ + ".pth")
            torch.save(model.state_dict(), ckpt_path)
            print("New best model saved!")

            with open(osp.join(results_dir, "best_result.json"), "w") as outfile:
                json.dump(
                    {
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "train_rec": train_rec,
                        "val_rec": val_rec,
                        "train_f1": train_f1,
                        "val_f1": val_f1,
                        "train_prec": train_prec,
                        "val_prec": val_prec,
                        "best_epoch": epoch,
                        "ckpt": ckpt_path,
                    },
                    outfile,
                    indent=2,
                )

            # Confusion matrix su VAL (best model)
            cm_val = confusion_matrix_torch(y_val, yp_val, num_classes=model.num_output)
            np.save(osp.join(results_dir, "cm_val.npy"), cm_val)
            save_confusion_matrix(
                cm_val,
                osp.join(results_dir, "cm_val.png"),
                class_names=[f"class_{i}" for i in range(model.num_output)],
                normalize=False,
            )
            save_confusion_matrix(
                cm_val,
                osp.join(results_dir, "cm_val_norm.png"),
                class_names=[f"class_{i}" for i in range(model.num_output)],
                normalize=True,
            )

            # Confusion matrix su TEST (best model)
            test_acc, test_prec, test_rec, test_f1, test_loss_sum, y_test, yp_test = test_classifier(
                model, test_loader, device, return_preds=True
            )
            cm_test = confusion_matrix_torch(y_test, yp_test, num_classes=model.num_output)
            np.save(osp.join(results_dir, "cm_test.npy"), cm_test)
            save_confusion_matrix(
                cm_test,
                osp.join(results_dir, "cm_test.png"),
                class_names=[f"class_{i}" for i in range(model.num_output)],
                normalize=False,
            )
            save_confusion_matrix(
                cm_test,
                osp.join(results_dir, "cm_test_norm.png"),
                class_names=[f"class_{i}" for i in range(model.num_output)],
                normalize=True,
            )

            # log test metrics del best in TB
            writer.add_scalar("Accuracy/test(best)", test_acc, epoch)
            writer.add_scalar("F1/test(best)", test_f1, epoch)

    # Alla fine: salva CSV “tabella valori”
    csv_path = osp.join(results_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(history[0].keys()) if history else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)

    print(f"[OK] Saved training table to: {csv_path}")
    print(f"[OK] Saved confusion matrices to: {results_dir}")


def train_epoch_regressor(model, train_loader, len_train, optimizer, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(output, data.y)

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train


def test_regressor(model, loader, len_loader, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(pred, data.y).detach()

        loss_all += data.num_graphs * loss.item()

    return loss_all / len_loader


def train_cycle_regressor(task, train_loader, val_loader, test_loader,
                          len_train, len_val, len_test, model,
                          optimizer, device, base_path, epochs):

    best_acc = (0, 0)
    writer = SummaryWriter(base_path + '/plots')

    best_error = (+10000, +10000)
    for epoch in range(epochs):
        loss = train_epoch_regressor(model, train_loader, len_train, optimizer, device)
        writer.add_scalar('Loss/train', loss, epoch)
        train_error = test_regressor(model, train_loader, len_train, device)
        val_error = test_regressor(model, val_loader, len_val, device)

        writer.add_scalar('MSE/train', train_error, epoch)
        writer.add_scalar('MSE/test', val_error, epoch)

        print(f'Epoch: {epoch}, Loss: {loss:.5f}')

        print(f'Training Error: {train_error:.5f}')
        print(f'Val Error: {val_error:.5f}')

        if best_error[1] > val_error:
            best_error = train_error, val_error
            torch.save(
                model.state_dict(),
                osp.join(base_path + '/ckpt/',
                         model.__class__.__name__ + ".pth")
            )
            print("New best model saved!")

            with open(base_path + '/best_result.json', 'w') as outfile:
                json.dump({'train_error': train_error,
                           'val_error': val_error}, outfile)
