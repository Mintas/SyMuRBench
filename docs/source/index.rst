************************
SyMuRBench documentation
************************
Quick start
===========
Installation
------------
First, you should clone this repo

.. code-block:: bash
   
   git clone https://github.com/Mintas/SyMuRBench.git

Then download datasets from huggingface

.. code-block:: bash

   .

and put them into project root directory.


Running benchmark with precomputed features
-------------------------------------------
Here is the exampe of running benchmark for 2 tasks: ComposerClassificationASAP and ScorePerformanceRetrievalASAP.
   by default, if no tasks provided, benchmark runs all tasks

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
      feature_extractors_list=[m21_pfe, js_pfe],
      tasks=[
         "ComposerClassificationASAP",
         "ScorePerformanceRetrievalASAP"
      ]
   )

   benchmark.run_all_tasks()
   benchmark.display_result()

Running benchmark, calculating and writing features to file
-----------------------------------------------------------
.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.feature_extractor import PersistentFeatureExtractor
   from symurbench.music21_extractor import Music21Extractor

   path_to_music21_features = "data/features/music21_full_dataset.parquet"

   m21_pfe = PersistentFeatureExtractor(
      feature_extractor=Music21Extractor(),
      persistence_path=path_to_music21_features,
      use_cached=False,
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

Running benchmark from config
-----------------------------
.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.music21_extractor import Music21Extractor

   config = {
      "ComposerClassificationASAP": {
         "metadata_csv_path":"data/datasets/composer_and_retrieval_datasets/metadata_composer_dataset.csv",
         "files_dir_path":"data/datasets/composer_and_retrieval_datasets/"
      }
   }

   m21_fe = Music21Extractor()

   benchmark = Benchmark.init_from_config(
      feature_extractors_list=[m21_fe],
      tasks_config=config
   )
   benchmark.run_all_tasks()
   benchmark.display_result()

Running benchmark from YAML file with config
--------------------------------------------
.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.music21_extractor import Music21Extractor

   config_path = "configs/tasks_config.yaml"

   m21_fe = Music21Extractor()

   benchmark = Benchmark.init_from_config_file(
      feature_extractors_list=[m21_fe],
      tasks_config_path=config_path
   )
   benchmark.run_all_tasks()
   benchmark.display_result()

Saving pandas DataFrame with results
------------------------------------
.. code-block:: python

   from symurbench.benchmark import Benchmark
   from symurbench.music21_extractor import Music21Extractor

   path_to_music21_features = "data/features/music21_full_dataset.parquet"

   m21_pfe = PersistentFeatureExtractor(
      feature_extractor=Music21Extractor(),
      persistence_path=path_to_music21_features,
      use_cached=False,
      name="music21"
   )

   benchmark = Benchmark.init_from_config_file(
      feature_extractors_list=[m21_pfe]
   )
   benchmark.run_all_tasks()
   benchmark.get_result_df(round_num=3, add_std=True).to_csv("result.csv")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   modules

Indices and tables
==================


* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
