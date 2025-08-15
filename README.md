<p align="center">
  <img width="300" src="docs/assets/logo.jpg"/>
</p>

<h1 align="center"><i>SyMuRBench</i></h1>
<p align="center"><i>Benchmark for Symbolic Music Representations</i></p>

[![GitHub Release](https://img.shields.io/github/v/release/Mintas/SyMuRBench)](https://pypi.python.org/pypi/symurbench/)
[![GitHub License](https://img.shields.io/github/license/Mintas/SyMuRBench)](https://github.com/Mintas/SyMuRBench/blob/main/LICENSE)

## 1. Overview

SyMuRBench is a versatile benchmark designed to compare vector representations of symbolic music. We provide standardized test splits from well-known datasets and strongly encourage authors to **exclude files from these splits** when training models to ensure fair evaluation. Additionally, we introduce a novel **score-performance retrieval task** to evaluate the alignment between symbolic scores and their performed versions.

## 2. Tasks Description

| Task Name                     | Source Dataset | Task Type               | # of Classes | # of Files       | Default Metrics                                  |
|-------------------------------|--------------|--------------------------|-------------|------------------|--------------------------------------------------|
| ComposerClassificationASAP    | ASAP         | Multiclass Classification | 7           | 197              | Weighted F1 Score, Balanced Accuracy             |
| GenreClassificationMMD        | MetaMIDI     | Multiclass Classification | 7           | 2,795            | Weighted F1 Score, Balanced Accuracy             |
| GenreClassificationWMTX       | WikiMT-X     | Multiclass Classification | 8           | 985              | Weighted F1 Score, Balanced Accuracy             |
| EmotionClassificationEMOPIA   | Emopia       | Multiclass Classification | 4           | 191              | Weighted F1 Score, Balanced Accuracy             |
| EmotionClassificationMIREX    | MIREX        | Multiclass Classification | 5           | 163              | Weighted F1 Score, Balanced Accuracy             |
| InstrumentDetectionMMD        | MetaMIDI     | Multilabel Classification | 128         | 4,675            | Weighted F1 Score                                |
| ScorePerformanceRetrievalASAP | ASAP         | Retrieval                 | -           | 438 (219 pairs)  | R@1, R@5, R@10, Median Rank                      |

> **Note**: "ScorePerformanceRetrievalASAP" evaluates how well a model retrieves the correct performed version given a symbolic score (and vice versa), using paired score-performance MIDI files.

---

## 3. Baseline Features

As baselines, we provide precomputed features from [**music21**](https://github.com/cuthbertLab/music21) and [**jSymbolic2**](https://github.com/DDMAL/jSymbolic2). A `FeatureExtractor` for music21 is available in `src/symurbench/music21_extractor.py`.

---

## 4. Installation

Install the package via pip:

```bash
pip install symurbench
```

Then download the datasets and (optionally) precomputed features:

```python
from symurbench.utils import load_datasets

output_folder = "symurbench_data"     # Absolute or relative path to save data
load_datasets(
    output_folder=output_folder,
    load_features=True                # Downloads precomputed music21 & jSymbolic features
)
```

---

## 4. Usage Examples.

**Example 1: Using Precomputed Features**

Run benchmark on specific tasks using cached music21 and jSymbolic features.

```python
from symurbench.benchmark import Benchmark
from symurbench.feature_extractor import PersistentFeatureExtractor

path_to_music21_features = "symurbench_data/features/music21_full_dataset.parquet"
path_to_jsymbolic_features = "symurbench_data/features/jsymbolic_full_dataset.parquet"

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
    tasks=[ # By default, if no specific tasks are specified, the benchmark will run all tasks.
        "ComposerClassificationASAP",
        "ScorePerformanceRetrievalASAP"
    ]
)

benchmark.run_all_tasks()
benchmark.display_result(return_ci=True, alpha=0.05)
```

> **Tip**: If tasks is omitted, all available tasks will be run by default.

*Output Example*

![output](docs/assets/example.png?raw=true "")


**Example 2: Using a Configuration Dictionary**

Run benchmark with custom dataset paths and AutoML configuration.

```python
from symurbench.benchmark import Benchmark
from symurbench.music21_extractor import Music21Extractor
from symurbench.constant import DEFAULT_LAML_CONFIG_PATHS # dict with paths to AutoML configs

multiclass_task_automl_cfg_path = DEFAULT_LAML_CONFIG_PATHS["multiclass"]
print(f"AutoML config path: {multiclass_task_automl_cfg_path}")

config = {
    "ComposerClassificationASAP": {
        "metadata_csv_path":"symurbench_data/datasets/composer_and_retrieval_datasets/metadata_composer_dataset.csv",
        "files_dir_path":"symurbench_data/datasets/composer_and_retrieval_datasets/",
        "automl_config_path":multiclass_task_automl_cfg_path
    }
}

m21_fe = Music21Extractor()

benchmark = Benchmark.init_from_config(
    feature_extractors_list=[m21_fe],
    tasks_config=config
)
benchmark.run_all_tasks()
benchmark.display_result()
```

**Example 3: Using a YAML Configuration File**

Load task configurations from a YAML file (e.g., dataset paths, AutoML config paths).

```python
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
```

**Example 4: Saving Results to CSV**

Run benchmark and export results to a CSV file using pandas.

```python

from symurbench.benchmark import Benchmark
from symurbench.music21_extractor import Music21Extractor

path_to_music21_features = "symurbench_data/features/music21_features.parquet"

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
results_df = benchmark.get_result_df(round_num=3, return_ci=True)
results_df.to_csv("results.csv")
```

> **💡**: `round_num=3`: Round metrics to 3 decimal places.
`return_ci=True`: Include confidence intervals in the output.

## 6. Notes & Best Practices

- 🔒 **Avoid data leakage**: Do not include test-set files in your training data to ensure fair and valid evaluation.
- 🔄 **Reproducibility**: Use fixed random seeds and consistent preprocessing pipelines to make experiments reproducible.
- 📁 **File paths**: Ensure paths in config files are correct and accessible.
- 🧪 **Custom extractors**: You can implement your own `FeatureExtractor` subclass by inheriting from the base `FeatureExtractor` class and implementing the `extract` method.

## 7. Citation

If you use SyMuRBench in your research, please cite:

```bibtex
@inproceedings{symurbench2025,
  author    = {Petr Strepetov and Dmitrii Kovalev},
  title     = {SyMuRBench: Benchmark for Symbolic Music Representations},
  booktitle = {Proceedings of the 3rd International Workshop on Multimedia Content Generation and Evaluation: New Methods and Practice (McGE '25)},
  year      = {2025},
  pages     = {9},
  publisher = {ACM},
  address   = {Dublin, Ireland},
  doi       = {10.1145/3746278.3759392}
}
```
