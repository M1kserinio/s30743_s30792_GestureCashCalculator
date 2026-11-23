#Odpornosc: wszystkie potoki i klasyfikatory na kazdym folderze z datasets/odpornosc/

from objects.robustness.evaluator import RobustnessEvaluator

if __name__ == "__main__":
    RobustnessEvaluator().evaluate_all()
