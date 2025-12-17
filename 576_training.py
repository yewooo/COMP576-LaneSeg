import os
import pandas as pd
from ultralytics import YOLO

DATA_YAML = r"F:\COMP576\final\culane_6k\lane-seg.yaml"
ROOT_PROJECT_DIR = r"F:\COMP576\final\ablation_runs"

def find_results_csv(exp_dir: str) -> str | None:
    cands = [
        os.path.join(exp_dir, "results.csv"),
        os.path.join(exp_dir, "train", "results.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def main():
    os.makedirs(ROOT_PROJECT_DIR, exist_ok=True)

    configs = []

    configs.append({
        "name": "trial_1",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.08, "erasing": 0.25, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_2",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.05, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_3",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 1.0e-3, "scale": 0.3, "translate": 0.25,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_4",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 512, "batch": 4, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_5",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.1, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_6",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 1e-3,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_7",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 4, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_8",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 5e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_9",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 4, "weight_decay": 7e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    configs.append({
        "name": "trial_10",
        "model": "yolo11s-seg.pt",
        "train": {"epochs": 5, "imgsz": 640, "batch": 2, "weight_decay": 8e-4,
                  "mask_ratio": 2, "perspective": 8e-4, "scale": 0.2, "translate": 0.2,
                  "mixup": 0.0, "copy_paste": 0.05, "erasing": 0.15, "fliplr": 0.5, "flipud": 0.0}
    })

    resulting_content = []

    for i, cfg in enumerate(configs, start=1):
        exp_name = cfg["name"]
        print(f"\n===== Running {exp_name} ({i}/{len(configs)}) =====")

        cfg["train"]["workers"] = cfg["train"].get("workers", 0)
        cfg["train"]["device"] = 0

        model = YOLO(cfg["model"])

        model.train(
            data=DATA_YAML,
            project=ROOT_PROJECT_DIR,
            name=exp_name,
            exist_ok=True,
            **cfg["train"]
        )

        exp_dir = os.path.join(ROOT_PROJECT_DIR, exp_name)
        results_csv = find_results_csv(exp_dir)
        if not results_csv:
            print(f"[WARN] results.csv not found under: {exp_dir}")
            continue

        df = pd.read_csv(results_csv)
        last = df.iloc[-1]

        training_time = float(last["time"])
        seg_loss = float(last["val/seg_loss"])
        cls_loss = float(last["val/cls_loss"])

        map50_box = float(last["metrics/mAP50(B)"])
        map5095_box = float(last["metrics/mAP50-95(B)"])
        recall_box = float(last["metrics/recall(B)"])

        map50_seg = float(last["metrics/mAP50(M)"])
        map5095_seg = float(last["metrics/mAP50-95(M)"])
        recall_seg = float(last["metrics/recall(M)"])

        metric_map = 0.9 * (0.7 * map5095_seg + 0.3 * map50_seg) + 0.1 * (0.7 * map5095_box + 0.3 * map50_box)
        metric_recall = 0.9 * recall_seg + 0.1 * recall_box
        metric_stability = (0.9 * df["val/seg_loss"].std() + 0.1 * df["val/cls_loss"].std()) / (
            0.9 * df["val/seg_loss"].mean() + 0.1 * df["val/cls_loss"].mean() + 1e-6
        )
        metric_loss = 0.9 * seg_loss + 0.1 * cls_loss
        metric_time = training_time

        combined_metric = (0.4 * metric_map + 0.3 * metric_recall - 0.25 * metric_stability - 0.05 * metric_loss) / (
            metric_time + 1e-9
        )

        t = cfg["train"]
        resulting_content.append({
            "experiment": exp_name,
            "map": metric_map,
            "recall": metric_recall,
            "stability": metric_stability,
            "loss": metric_loss,
            "time": metric_time,
            "overview": combined_metric,

            "mask_ratio": t["mask_ratio"],
            "perspective": t["perspective"],
            "scale": t["scale"],
            "translate": t["translate"],
            "mixup": t["mixup"],
            "copy_paste": t["copy_paste"],
            "erasing": t["erasing"],
            "imgsz": t["imgsz"],
            "batch": t["batch"],
            "lr0": t.get("lr0", None),
            "weight_decay": t["weight_decay"],
            "workers": t.get("workers", None),
            "results_csv": results_csv,
        })

    summary_path = os.path.join(ROOT_PROJECT_DIR, "training_summary.csv")
    resulting_df = pd.DataFrame(resulting_content)
    resulting_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}\n")

    if len(resulting_df) == 0:
        print("No experiments produced results.csv")
        return

    print(resulting_df[["experiment", "map", "recall", "stability", "loss", "time", "overview"]])

    print("\nBest by each metric:")
    print("best_map:", resulting_df.loc[resulting_df["map"].idxmax(), "experiment"])
    print("best_recall:", resulting_df.loc[resulting_df["recall"].idxmax(), "experiment"])
    print("best_stability:", resulting_df.loc[resulting_df["stability"].idxmax(), "experiment"])
    print("best_loss:", resulting_df.loc[resulting_df["loss"].idxmin(), "experiment"])
    print("best_time:", resulting_df.loc[resulting_df["time"].idxmin(), "experiment"])
    print("best_overview:", resulting_df.loc[resulting_df["overview"].idxmax(), "experiment"])

if __name__ == "__main__":
    main()
