from objects.yolo_evaluator import YoloEvaluator

#wyniki trafiaja do results/, najlepszy model do models/pieniadze_best.pt
if __name__ == "__main__":
    YoloEvaluator().evaluate_all()
