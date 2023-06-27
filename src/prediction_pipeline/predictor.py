import logging
import os

import numpy as np
import torch.nn.functional as F
from bs4 import BeautifulSoup
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor(object):
    """Preprocessor takes in a text at a time and returns a preprocessed text."""

    def __init__(self):
        pass

    def preprocess(self, text: str):
        return BeautifulSoup(text).get_text()


class CustomPredictor(Predictor):
    """Custom Predictor for the prediction pipeline."""

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        """Loads the model artifact.
        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
        Raises:
            ValueError: If there's no required model files provided in the
            artifacts uri.
        """
        prediction_utils.download_model_artifacts(artifacts_uri)
        logger.info(os.listdir())

        if "model_artifacts" in os.listdir():
            model_artifacts_dir = "model_artifacts"
        else:
            model_artifacts_dir = "."

        logger.info(f"Model artifacts directory: {model_artifacts_dir}")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_artifacts_dir
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_artifacts_dir)
        self._labels = list(self._model.config.id2label.values())
        self._preprocessor = Preprocessor()

        logger.info(f"Number of candidates: {len(self._labels)}")
        logger.info(f"Candidates: {self._labels}")

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Preprocesses the prediction input.

        Args:
            prediction_input (dict):
                Required. The prediction input follows the format:
                { "instances": [ { "sequence": "text to classify" } ] }
        Returns:
            The preprocessed prediction input.
        """
        instances = prediction_input["instances"]
        logger.info(f"Number of instances: {len(instances)}")
        return [
            self._preprocessor.preprocess(instance["sequence"])
            for instance in instances
        ]

    def predict(self, instances: np.ndarray) -> np.ndarray:
        """Performs prediction.
        Args:
            instances:
                Required. The instance(s) used for performing prediction.
        Returns:
            The prediction results. The results follow the format:
            { "predictions": [ { "sequence": "text to classify",
            "labels": ["label1", "label2"], "scores": [0.9, 0.1] } ] }
        """
        predictions = []
        for instance in instances:
            result = {"sequence": instance}
            logits = self._model(
                **self._tokenizer(
                    instance,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
            ).logits
            scores = F.softmax(logits, dim=1)
            result["labels"] = self._labels
            result["scores"] = [prob.item() for prob in scores[0]]
            predictions.append(result)
        return predictions

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Converts numpy array to a dict, and sorts the labels and scores.
        Args:
            prediction_results:
                Required. The prediction results follow the format:
                { "predictions": [ { "sequence": "text to classify",
                "labels": ["label1", "label2"], "scores": [0.9, 0.1] } ] }
        Returns:
            The postprocessed prediction results.
        """
        return {"predictions": prediction_results}
