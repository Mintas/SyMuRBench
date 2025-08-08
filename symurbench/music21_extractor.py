"""FeatureExtractor for music21 features."""
import logging

import numpy as np
from music21.features.base import allFeaturesAsList, extractorsById

from .feature_extractor import FeatureExtractor

logger = logging.getLogger("music21 extractor")

categorical_features = {
    "R31_1": (0, 1, 2, 4, 8, 16, 32, 64, 128), # InitialTimeSignatureFeature (denom)
    "P6_0": tuple(range(12)), # IntervalBetweenStrongestPitchClassesFeature
    "P9_0": tuple(range(13)), # PitchClassVarietyFeature
    "P16_0": tuple(range(12)) # MostCommonPitchClassFeature
}

feats2del = ["TX1_0"]

class Music21Extractor(FeatureExtractor):
    """FeatureExtractor Subclass for extracting music21 features from MIDI files."""
    def __init__(
        self,
        extractor_name: str = "music21",
        fast: bool = True,
        preprocess_features: bool = True,
    ) -> None:
        """
        Initialize the Music21Extractor class.

        Args:
            extractor_name (str): Name of the feature extractor. Should be unique.
            fast (bool, optional):
                if True, will use multiprocessing to extract features.
                Defaults to False.
            preprocess_features (bool, optional):
                whether to preprocess features for tasks with AutoML.
                If True, preprocess features according to automl config.
                If False, use features as they are.
                Defaults to True.
        """
        super().__init__(extractor_name, fast, preprocess_features)

    def extract_features_from_file(
        self,
        file: str
    ) -> np.ndarray:
        """
        Extract all music21 features from MIDI file.

        Args:
            file (str): relative path to MIDI file

        Raises:
            ValueError: if an error occurred while extracting features from the file.

        Returns:
            np.ndarray:
                numpy array with shape (n_features,) with extracted music21 features
        """
        try:
            features = allFeaturesAsList(file)
        except Exception as e:  # noqa: BLE001
            log_msg = f"Cannot extract features from file {file}. {e}"
            msg = f"Exception during feature extraction: {e}"
            logger.error(log_msg)
            raise ValueError(msg) from None
        columns = [x.id for x in extractorsById("all")]
        unique_feats = {
            (columns[outer] + f"_{i}"): f
            for outer in range(len(columns))
            for i, f in enumerate(features[outer])
        }

        # OHE for categorical features
        for feat in categorical_features:
            value = unique_feats[feat]
            del unique_feats[feat]
            for cat in categorical_features[feat]:
                unique_feats[f"{feat}_{cat}"] = 0 if cat != value else 1

        # delete text features
        for feat in feats2del:
            del unique_feats[feat]

        return np.array(list(unique_feats.values()))
