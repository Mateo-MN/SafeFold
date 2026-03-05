import joblib
import warnings
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

#Our model was trained using a different 
warnings.filterwarnings("ignore")


pipeline = joblib.load(Path(__file__).parent / "./Toxicity_prediction/tox_pred.joblib")

def go_terms_to_toxicity(GO_terms):
    X = pd.DataFrame([GO_terms])

    cols = pipeline.feature_names_in_

    X = X.reindex(columns=cols, fill_value=0)

    score = pipeline.predict_proba(X)[0, 1]
    return score

if __name__ == "__main__":
    go_terms = {
        "GO:0005488": 0.54,
        "GO:0005515": 0.27,
        "GO:0030234": 0.25
    }

    print(go_terms_to_toxicity(go_terms))
