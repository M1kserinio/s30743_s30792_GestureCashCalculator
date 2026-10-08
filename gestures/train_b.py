from train import run

EVAL = "kfold"  # "lopo" = trening na 5 osobach, test na szostej; "kfold" = 5-fold

# B: uczenie maszynowe na surowych, znormalizowanych landmarkach
run("B", use_features=False, eval=EVAL)