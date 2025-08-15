Running Benchmark with Feature Caching
======================================

This example demonstrates how to run the benchmark while extracting features on-the-fly and caching them to disk using :class:`~symurbench.feature_extractor.PersistentFeatureExtractor`.

The features (in this case from ``music21``) are computed during execution and saved to the file specified by ``persistence_path``. This avoids recomputation in future runs when ``use_cached=True``.

.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.feature_extractor import PersistentFeatureExtractor
   from symurbench.music21_extractor import Music21Extractor

   path_to_music21_features = "data/features/music21_full_dataset.parquet"

   m21_pfe = PersistentFeatureExtractor(
       feature_extractor=Music21Extractor(),
       persistence_path=path_to_music21_features,
       use_cached=False,  # Compute features now and save to file
       name="music21"
   )

   benchmark = Benchmark(
       feature_extractors_list=[m21_pfe],
       tasks=[
           "ComposerClassificationASAP",
           "ScorePerformanceRetrievalASAP"
       ]
   )

   benchmark.run_all_tasks()
   benchmark.display_result()