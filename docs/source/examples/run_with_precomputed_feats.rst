Running Benchmark with Precomputed Features
===========================================

Here is an example of running the benchmark for two tasks: ``ComposerClassificationASAP`` and ``ScorePerformanceRetrievalASAP``.
By default, if no tasks are provided, the benchmark runs all available tasks.

.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.feature_extractor import PersistentFeatureExtractor

   path_to_music21_features = "data/features/music21_full_dataset.parquet"
   path_to_jsymbolic_features = "data/features/jsymbolic_full_dataset.parquet"

   m21_pfe = PersistentFeatureExtractor(
      persistence_path=path_to_music21_features,
      use_cached=True,
      name="music21"
   )
   jsymb_pfe = PersistentFeatureExtractor(
      persistence_path=path_to_jsymbolic_features,
      use_cached=True,
      name="jSymbolic"
   )

   benchmark = Benchmark(
      feature_extractors_list=[m21_pfe, jsymb_pfe],
      tasks=[
         "ComposerClassificationASAP",
         "ScorePerformanceRetrievalASAP"
      ]
   )

   benchmark.run_all_tasks()
   benchmark.display_result()
