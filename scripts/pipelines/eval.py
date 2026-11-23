#Porownanie potokow: sam detektor kontra lokalizator + klasyfikator, na tych samych zdjeciach

from objects.pipelines.evaluator import PipelineEvaluator

if __name__ == "__main__":
    PipelineEvaluator().evaluate_all()
