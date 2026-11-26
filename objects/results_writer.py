import csv
import os
import platform

from objects.config import RESULTS_DIR, benchmark_devices, device_name


class ResultsWriter:
    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)

    def save_csv(self, file_name, rows):
        with open(RESULTS_DIR / file_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Zapisano {RESULTS_DIR / file_name}")

    def save_hardware(self):
        import torch
        import ultralytics

        accelerator = "brak"
        if torch.cuda.is_available():
            accelerator = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            accelerator = "Apple MPS"
        self.save_csv("study2_hardware.csv", [{
            "system": platform.platform(),
            "procesor": platform.processor() or platform.machine(),
            "rdzenie_cpu": os.cpu_count(),
            "akcelerator": accelerator,
            "urzadzenia_w_tabelach": " ".join(device_name(d) for d in benchmark_devices()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            #bez MKLDNN splot depthwise (MobileNet) liczy sie na CPU kilka razy wolniej niz powinien
            "torch_mkldnn": torch.backends.mkldnn.is_available(),
            "ultralytics": ultralytics.__version__,
        }])

    def plt(self):
        import matplotlib
        matplotlib.use("Agg")  #wykres tylko do pliku, bez otwierania okna
        import matplotlib.pyplot as plt
        return plt

    def save_map_vs_fps_plot(self, summary, device_names, file_name):
        plt = self.plt()

        #najlepszy model w prawym gornym rogu
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in device_names:
            fps = [row[f"fps_{name}"] for row in summary]
            maps = [row["test_map50_95"] for row in summary]
            ax.plot(fps, maps, "o", label=name)
            for row, x, y in zip(summary, fps, maps):
                ax.annotate(row["model"], (x, y), textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("FPS")
        ax.set_ylabel("mAP50-95 (test)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / file_name, dpi=200)
        plt.close(fig)
        print(f"Zapisano {RESULTS_DIR / file_name}")

    def save_confusion_plot(self, file_name, matrices):
        #matrices: nazwa modelu -> (nazwy klas, macierz [prawdziwa][przewidziana])
        plt = self.plt()
        columns = min(3, len(matrices))
        rows = (len(matrices) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (name, (labels, matrix)) in zip(axes.flat, matrices.items()):
            ax.axis("on")
            ax.imshow(matrix, cmap="Blues")
            ax.set_title(name, fontsize=10)
            ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("przewidziane", fontsize=8)
            ax.set_ylabel("prawdziwe", fontsize=8)
            biggest = max(max(row) for row in matrix) or 1
            for i, row in enumerate(matrix):
                for j, count in enumerate(row):
                    if count:
                        color = "white" if count > biggest / 2 else "black"
                        ax.text(j, i, count, ha="center", va="center", fontsize=7, color=color)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / file_name, dpi=200)
        plt.close(fig)
        print(f"Zapisano {RESULTS_DIR / file_name}")

    def save_heatmap(self, file_name, title, row_names, column_names, values):
        #tabela odpornosci jako obrazek, 0 (czerwony) - 1 (zielony)
        plt = self.plt()
        fig, ax = plt.subplots(figsize=(1.2 * len(column_names) + 2.5, 0.38 * len(row_names) + 1.8))
        ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(column_names)), column_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_names)), row_names, fontsize=8)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / file_name, dpi=200)
        plt.close(fig)
        print(f"Zapisano {RESULTS_DIR / file_name}")
