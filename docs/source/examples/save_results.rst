Exporting Benchmark Results to CSV
==================================

After running the benchmark, you can export the results to a CSV file using pandas for further analysis, reporting, or sharing.

This example demonstrates how to:
- Run the benchmark with feature caching.
- Retrieve results as a :class:`pandas.DataFrame`.
- Include confidence intervals (optional).
- Save the formatted results to a CSV file.

.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.feature_extractor import PersistentFeatureExtractor
   from symurbench.music21_extractor import Music21Extractor

   # Configure feature extraction with caching
   path_to_music21_features = "symurbench_data/features/music21_features.parquet"

   m21_pfe = PersistentFeatureExtractor(
       feature_extractor=Music21Extractor(),
       persistence_path=path_to_music21_features,
       use_cached=False,
       name="music21"
   )

   # Initialize benchmark from config file
   benchmark = Benchmark.init_from_config_file(
       feature_extractors_list=[m21_pfe]
   )

   # Run all tasks
   benchmark.run_all_tasks()

   # Get results as DataFrame (rounded, with confidence intervals)
   results_df = benchmark.get_result_df(round_num=3, return_ci=True)

   # Export to CSV
   results_df.to_csv("results.csv")

   print("Results saved to results.csv")
