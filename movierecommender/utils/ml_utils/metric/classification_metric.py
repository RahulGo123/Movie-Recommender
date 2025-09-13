import sys
from movierecommender.entity.artifact_entity import ClassificationMetricArtifact
from movierecommender.exception.exception import MovieRecommenderException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def get_classification_score(y_true, y_pred):
    try:
        return {
        "f1_score_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_score_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
        }
    except Exception as e:
        raise MovieRecommenderException(e, sys)