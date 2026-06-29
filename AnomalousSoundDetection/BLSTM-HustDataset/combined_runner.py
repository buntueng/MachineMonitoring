"""
combined_runner.py  —  Combined LSTM / BiLSTM / Solve_BiLSTM trainer
=====================================================================
Faithful combination of the 3 original scripts — zero logic changes.

Run commands:
    python3 combined_runner.py --model lstm         --data_dir /path/to/data
    python3 combined_runner.py --model bilstm        --data_dir /path/to/data
    python3 combined_runner.py --model solve_bilstm  --data_dir /path/to/data

Run all 3 at once (Linux/Mac):
    for model in lstm bilstm solve_bilstm; do
        python3 combined_runner.py --model $model --data_dir "/your/data/path"
    done
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow.keras import callbacks, layers, models
from tqdm import tqdm


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["lstm", "bilstm", "solve_bilstm"])
    parser.add_argument("--data_dir",
                        default=r"/home/eecommu06/Documents/Jin/motor_detection/New1")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


OUTPUT_SUBDIR = {
    "lstm":         "History/LSTM/200k",
    "bilstm":       "History/BiLSTM/200k",
    "solve_bilstm": "History/Solve_BiLSTM_Fast2/200k",
}


# =========================
# GPU SETUP
# =========================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], "GPU")
            print(f"[GPU] Using: {gpus[0].name}")
        except RuntimeError as e:
            print(f"[GPU] Setup error: {e}")
    else:
        print("[GPU] No CUDA device found — running on CPU.")


# =========================
# CALLBACKS  (identical in all 3 originals)
# =========================
class TQDMProgressBar(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]
        self.pbar_epoch = tqdm(total=self.epochs, desc="Total Epochs", position=0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        steps = self.params.get("steps", None)
        self.pbar_batch = (
            tqdm(total=steps, desc=f"Epoch {epoch+1}/{self.epochs}",
                 position=1, leave=False)
            if steps is not None else None
        )

    def on_batch_end(self, batch, logs=None):
        if self.pbar_batch:
            self.pbar_batch.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar_batch:
            self.pbar_batch.close()
        self.pbar_epoch.update(1)

    def on_train_end(self, logs=None):
        self.pbar_epoch.close()


class MetricsLogger(callbacks.Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.rows = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.rows.append({
            "epoch":          epoch + 1,
            "loss":           logs.get("loss"),
            "val_loss":       logs.get("val_loss"),
            "accuracy":       logs.get("accuracy"),
            "val_accuracy":   logs.get("val_accuracy"),
            "learning_rate":  lr,
            "epoch_time_sec": epoch_time,
        })
        pd.DataFrame(self.rows).to_csv(self.file_path, index=False)


# =========================
# MODEL FACTORIES  (exact copy from each original)
# =========================
def create_lstm_model(num_classes):
    """Exact copy of LSTM200k.py create_model()"""
    inputs = layers.Input(shape=(2048, 1))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_bilstm_model(num_classes):
    """Exact copy of BiLSTM200k.py create_model()"""
    inputs = layers.Input(shape=(2048, 1))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_solve_bilstm_model(num_classes, seq_len=2048):
    """Exact copy of Solve_BiLSTM200k.py create_model()"""
    inputs = layers.Input(shape=(seq_len, 1), name="input")

    stride = 32
    T_red  = seq_len // stride   # 64 timesteps

    x = layers.Reshape((T_red, stride), name="fold_time")(inputs)
    x = layers.Dense(16, activation="relu", name="linear_proj")(x)

    lstm_out = layers.Bidirectional(
        layers.LSTM(4, return_sequences=True),
        merge_mode="sum",
        name="bilstm",
    )(x)

    attn_score  = layers.Dense(1, name="attn_score")(lstm_out)
    attn_weight = layers.Softmax(axis=1, name="attn_softmax")(attn_score)
    context     = layers.Dot(axes=1, name="attn_pool")([attn_weight, lstm_out])
    x           = layers.Flatten(name="flatten_context")(context)

    x       = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Fast3_Keep64")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_FACTORIES = {
    "lstm":         create_lstm_model,
    "bilstm":       create_bilstm_model,
    "solve_bilstm": create_solve_bilstm_model,
}

# LSTM & BiLSTM monitor val_accuracy (max); Solve_BiLSTM monitors val_loss (min)
CHECKPOINT_CFG = {
    "lstm":         ("val_accuracy", "max"),
    "bilstm":       ("val_accuracy", "max"),
    "solve_bilstm": ("val_loss",     "min"),
}

# LSTM & BiLSTM use test_acc from evaluate(); Solve_BiLSTM uses accuracy_score()
USE_EVAL_ACC = {
    "lstm":         True,
    "bilstm":       True,
    "solve_bilstm": False,
}


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        args.data_dir, OUTPUT_SUBDIR[args.model]
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Model     : {args.model.upper()}")
    print(f"[INFO] Data dir  : {args.data_dir}")
    print(f"[INFO] Output dir: {output_dir}\n")

    setup_gpu()

    # ---- Data (identical in all 3 originals) ----
    data   = np.load(os.path.join(args.data_dir, "balanced_data.npy"),   mmap_mode="r")
    labels = np.load(os.path.join(args.data_dir, "balanced_labels.npy"), mmap_mode="r")

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    num_classes = len(np.unique(labels))

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Print model summary once
    tmp = MODEL_FACTORIES[args.model](num_classes)
    tmp.summary()
    del tmp

    # ---- Summary CSV ----
    summary_file = os.path.join(output_dir, "all_fold_metrics.csv")
    pd.DataFrame(columns=[
        "Fold", "Train_Time", "Test_Time", "Test_Loss",
        "Accuracy", "Precision", "Recall", "F1", "AUC"
    ]).to_csv(summary_file, index=False)

    monitor, mode = CHECKPOINT_CFG[args.model]
    use_eval_acc  = USE_EVAL_ACC[args.model]

    # ---- 10-Fold CV ----
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    overall_start = time.time()

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(X_train_val, y_train_val), start=1),
        total=10, desc="Overall Progress"
    ):
        print(f"\n===== Fold {fold}/10 =====")

        model    = MODEL_FACTORIES[args.model](num_classes)
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        best_model_path = os.path.join(fold_dir, f"best_model_fold{fold}.keras")
        history_csv     = os.path.join(fold_dir, f"history_fold{fold}.csv")

        cb = [
            callbacks.ModelCheckpoint(
                best_model_path, monitor=monitor, mode=mode,
                save_best_only=True, verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=3, factor=0.5, verbose=1,
            ),
            MetricsLogger(history_csv),
            TQDMProgressBar(),
        ]

        train_start = time.time()
        model.fit(
            X_train_val[train_idx], y_train_val[train_idx],
            validation_data=(X_train_val[val_idx], y_train_val[val_idx]),
            epochs=30, batch_size=64, verbose=0, callbacks=cb,
        )
        train_time = time.time() - train_start

        best_model          = tf.keras.models.load_model(best_model_path)
        test_start          = time.time()
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        y_prob              = best_model.predict(X_test, verbose=0)
        y_pred              = np.argmax(y_prob, axis=1)
        test_time           = time.time() - test_start

        # LSTM/BiLSTM use test_acc from evaluate(); Solve_BiLSTM uses accuracy_score()
        acc = test_acc if use_eval_acc else accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.save(os.path.join(fold_dir, f"cm_fold{fold}.npy"), cm)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.title(f"Confusion Matrix Fold {fold}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f"cm_fold{fold}.png"))
        plt.close()

        # ROC / AUC
        y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
        auc_score  = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
        plt.figure(figsize=(6, 5))
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob[:, c])
            plt.plot(fpr, tpr, label=f"Class {c}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Fold {fold}")
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f"roc_fold{fold}.png"))
        plt.close()

        pd.DataFrame([{
            "Fold": fold, "Train_Time": train_time, "Test_Time": test_time,
            "Test_Loss": test_loss, "Accuracy": acc, "Precision": precision,
            "Recall": recall, "F1": f1, "AUC": auc_score,
        }]).to_csv(summary_file, mode="a", header=False, index=False)

        print(f"Fold {fold} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc_score:.4f}")

    # ---- Wrap-up ----
    overall_time = time.time() - overall_start
    with open(os.path.join(output_dir, "total_training_time.txt"), "w") as f:
        f.write(f"Total Time (sec): {overall_time:.2f}\n")
        f.write(f"Total Time (hrs): {overall_time/3600:.3f}\n")

    print("\nTraining Completed")
    print(f"Total Time: {overall_time:.2f} sec  ({overall_time/3600:.3f} hrs)")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()