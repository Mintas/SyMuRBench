Running Benchmark from a YAML Configuration File
================================================

This example demonstrates how to configure and run the benchmark using an external YAML file. This approach is ideal for managing complex or reusable configurations without hardcoding paths and settings in Python.

.. code-block:: python

   from symurbench.benchmark import Benchmark
    from symurbench.music21_extractor import Music21Extractor
    from symurbench.constant import DATASETS_CONFIG_PATH # path to config with datasets paths

    print(f"Datasets config path: {DATASETS_CONFIG_PATH}")

    m21_fe = Music21Extractor()

    benchmark = Benchmark.init_from_config_file(
        feature_extractors_list=[m21_fe],
        tasks_config_path=DATASETS_CONFIG_PATH
    )
    benchmark.run_all_tasks()
    benchmark.display_result()
