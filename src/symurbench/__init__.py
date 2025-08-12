"""Init."""
from . import (
    benchmark,
    constant,
    feature_extractor,
    music21_extractor,
    retrieval,
    utils,
)
from .abstract_tasks import abstask, classification_task, retrieval_task
from .metaloaders import (
    metaloader,
    metaloader_multiclass,
    metaloader_multilabel,
    metaloader_retrieval,
)
from .metrics import metric_value, retrieval_scorer, scorer, sklearn_scorer
from .tasks import (
    ComposerClassificationASAP,
    EmotionClassificationEMOPIA,
    EmotionClassificationMIREX,
    GenreClassificationMMD,
    GenreClassificationWMTX,
    InstrumentDetectionMMD,
    ScorePerformanceRetrievalASAP,
)

__all__ = [
    "benchmark",
    "constant",
    "feature_extractor",
    "music21_extractor",
    "retrieval",
    "utils",
    "abstask",
    "classification_task",
    "retrieval_task",
    # Metaloaders
    "metaloader",
    "metaloader_multiclass",
    "metaloader_multilabel",
    "metaloader_retrieval",
    "metric_value",
    # Scorers
    "scorer",
    "retrieval_scorer",
    "sklearn_scorer",
    # Tasks
    "ComposerClassificationASAP",
    "EmotionClassificationEMOPIA",
    "EmotionClassificationMIREX",
    "GenreClassificationMMD",
    "GenreClassificationWMTX",
    "InstrumentDetectionMMD",
    "ScorePerformanceRetrievalASAP"
]
