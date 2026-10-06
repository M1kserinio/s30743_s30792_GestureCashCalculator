import csv

from objects.config import RESULTS_DIR


class ResultsWriter:
    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)

    def save_csv(self, file_name, rows):
        #rows = lista slownikow, klucze z pierwszego wiersza to naglowki kolumn
        with open(RESULTS_DIR / file_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Zapisano {RESULTS_DIR / file_name}")

    def save_map_vs_fps_plot(self, summary, device_names):
        #importuje tutaj, bo i tak reszta nie potzrebuhje tego importa a i tak sie laduje mega
        import matplotlib
        matplotlib.use("Agg")  #wykres tylko do pliku, bez otwierania okna
        import matplotlib.pyplot as plt

        #dokladnosc vs szybkosc - najlepszy model jest w prawym gornym rogu
        fig, ax = plt.subplots(figsize=(6, 4))
        for device_name in device_names:
            fps = [row[f"fps_{device_name}"] for row in summary]
            maps = [row["test_map50_95"] for row in summary]
            ax.plot(fps, maps, "o-", label=device_name)
            for row, x, y in zip(summary, fps, maps):
                ax.annotate(row["model"], (x, y), textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("FPS")
        ax.set_ylabel("mAP50-95 (test)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "study2_map_vs_fps.png", dpi=200)
        print(f"Zapisano {RESULTS_DIR / 'study2_map_vs_fps.png'}")
