Using a Configuration Dictionary
================================

This example shows how to run the benchmark using a custom configuration dictionary, allowing you to specify dataset paths and AutoML configurations per task.

You can also override the default AutoML pipeline settings using custom configuration files.

.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.music21_extractor import Music21Extractor
   from symurbench.constant import DEFAULT_LAML_CONFIG_PATHS  # Predefined AutoML config paths

   # Load AutoML configuration for multiclass classification tasks
   multiclass_task_automl_cfg_path = DEFAULT_LAML_CONFIG_PATHS["multiclass"]
   print(f"AutoML config path: {multiclass_task_automl_cfg_path}")

   # Define task-specific configuration
   config = {
       "ComposerClassificationASAP": {
           "metadata_csv_path": "symurbench_data/datasets/composer_and_retrieval_datasets/metadata_composer_dataset.csv",
           "files_dir_path": "symurbench_data/datasets/composer_and_retrieval_datasets/",
           "automl_config_path": multiclass_task_automl_cfg_path  # Optional: override default AutoML settings
       }
   }

   # Initialize feature extractor
   m21_fe = Music21Extractor()

   # Create benchmark from config
   benchmark = Benchmark.init_from_config(
       feature_extractors_list=[m21_fe],
       tasks_config=config
   )

   benchmark.run_all_tasks()
   benchmark.display_result()
