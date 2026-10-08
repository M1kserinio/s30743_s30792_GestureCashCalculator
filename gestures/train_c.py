from train import run

EVAL = "kfold"  # "lopo" = trening na 5 osobach, test na szostej; "kfold" = 5-fold

# C: uczenie maszynowe na cechach inzynieryjnych (features.py)
run("C", use_features=True, eval=EVAL)