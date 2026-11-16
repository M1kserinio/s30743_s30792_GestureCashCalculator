from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "gest_results"  # tu leżą pliki z wynikami

# Wczytanie wszystkich plików z wynikami
tables = []
for file in sorted(RESULTS_DIR.glob("*_results_lopo.csv")):
    table = pd.read_csv(file)
    table["method"] = file.name[0]  # pierwsza litera pliku: B, C albo D
    tables.append(table)
results = pd.concat(tables, ignore_index=True)  # wszystkie tabele w jednej

#  Nazwa modelu do wykresów
results["name"] = results["method"] + "-" + results["model"]
results = results.sort_values("accuracy", ascending=False)  # najlepszy model na górze
results = results.set_index("name")
results.to_csv(RESULTS_DIR / "comparison.csv")

# Tabela w konsoli
table_columns = ["accuracy", "f1", "person_mean", "person_std", "person_min", "latency"]
print(results[table_columns].round(3).to_string())

# Kolumny z wynikiem każdej osoby
person_columns = []
person_names = []
for column in results.columns:
    if column.startswith("acc_"):
        person_columns.append(column)
        person_names.append(column.replace("acc_", ""))

# 5. Wykresy
fig, ((ax_scores, ax_per_person), (ax_latency, ax_tradeoff)) = plt.subplots(2, 2, figsize=(15, 10))

# Wykres 1: accuracy i F1 każdego modelu
results[["accuracy", "f1"]].plot.bar(ax=ax_scores, ylim=(0, 1))
ax_scores.set_title("Accuracy and F1 score (leave-one-person-out)")
ax_scores.set_xlabel("Model (method letter - classifier)")
ax_scores.set_ylabel("Score (0 = worst, 1 = best)")

# Wykres 2: accuracy dla każdej osoby
heatmap = ax_per_person.imshow(results[person_columns].to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax_per_person.set_xticks(range(len(person_names)), person_names)
ax_per_person.set_yticks(range(len(results)), results.index)
ax_per_person.set_title("Accuracy for each test person")
ax_per_person.set_xlabel("Person used as the test set")
ax_per_person.set_ylabel("Model")
fig.colorbar(heatmap, ax=ax_per_person, label="Accuracy")

# Wykres 3: czas jednej predykcji
results["latency"].plot.barh(ax=ax_latency, logx=True)
ax_latency.set_title("Time of one prediction (shorter is better)")
ax_latency.set_xlabel("Latency [ms] (log scale)")
ax_latency.set_ylabel("Model")

# Wykres 4: dokładność kontra czas
ax_tradeoff.scatter(results["latency"], results["accuracy"])
for model_name, row in results.iterrows():
    ax_tradeoff.annotate(model_name, (row["latency"], row["accuracy"]), xytext=(4, 4), textcoords="offset points")
ax_tradeoff.set_xscale("log")
ax_tradeoff.set_title("Accuracy vs latency (top-left is best)")
ax_tradeoff.set_xlabel("Latency [ms] (log scale)")
ax_tradeoff.set_ylabel("Accuracy")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "comparison.png", dpi=150)  # zapis wykresów