# SyMuRBench: Benchmark for symbolic music representations


**1. Overview.**

SyMuRBench is a versatile benchmark designed to compare vector representations of symbolic music. With benchmark code we provide test splits for well-known datasets and encourage authors to not include files from these splits in their training data. Also we propose novel score-perfomance retrieval task for evaluating representations.

**2. Tasks description.**

| Task name | Source dataset | Task type | # of classes | # of files | Default metrics |
| -------- | ------- | ---------- | -------------- | ------- | ------- |
| ComposerClassificationASAP | ASAP | Multiclass classification | 7 | 197 | weighted f1 score, balanced accuracy |
| GenreClassificationMMD | MetaMIDI | Multiclass classification | 7 | 2795 | weighted f1 score, balanced accuracy |
| GenreClassificationWMTX | WikiMT-X | Multiclass classification | 8 | 985 | weighted f1 score, balanced accuracy |
| EmotionClassificationEMOPIA | Emopia | Multiclass classification | 4 | 191 | weighted f1 score, balanced accuracy |
| EmotionClassificaitonMIREX | MIREX | Multiclass classification | 5 | 163 | weighted f1 score, balanced accuracy |
| InstrumentDetectionMMD | MetaMIDI | Multilabel classification | 128 | 4675 | weighted f1 score |
| ScorePerfomanceRetrievalASAP | ASAP | Retrieval | - | 438 (219 pairs) | R@1, R@5, R@10, Median Rank |

**3. Baseline.**

As an example we provide precomputed music21 and jSymbolic2 features. Also we provide FeatureExtractor for music21 in music21_extractor.py

**4. Installation.**

Firstly, install the symurbench library using pip:

```
pip install symurbench
```

Then load datasets from huggingface:

```
from symurbench.utils import load_datasets


output_folder = "symurbench_data"
load_datasets(
    output_folder=output_folder, # Absolute or relative path to the target folder. The dataset and features will be extracted here. 
    load_features=True, # Whether do download precomputed music21 and jsymbolic features.
)
```

Thats all, now you can run the benchmark, here is an example with precomputed features:

**Example 1:** Executing the benchmark using precomputed features for the tasks "ComposerClassificationASAP" and "ScorePerformanceRetrievalASAP."

```
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
    feature_extractors_list=[m21_pfe, js_pfe],
    tasks=[ # By default, if no specific tasks are specified, the benchmark will run all tasks.
        "ComposerClassificationASAP",
        "ScorePerformanceRetrievalASAP"
    ]
)

benchmark.run_all_tasks()
benchmark.display_result(return_ci=True, alpha=0.05)
```


**4. Basic usage examples.**


**Example 2:** Executing the benchmark using a config
```
from symurbench.benchmark import Benchmark
from symurbench.music21_extractor import Music21Extractor
from symurbench.constant import DEFAULT_LAML_CONFIG_PATHS # dict with paths to AutoML configs

multiclass_task_automl_cfg_path = DEFAULT_LAML_CONFIG_PATHS["multiclass"]
print(f"AutoML config path: {multiclass_task_automl_cfg_path}")

config = {
    "ComposerClassificationASAP": {
        "metadata_csv_path":"data/datasets/composer_and_retrieval_datasets/metadata_composer_dataset.csv",
        "files_dir_path":"data/datasets/composer_and_retrieval_datasets/",
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

**Example 3:** Executing the benchmark using a YAML file containing the configuration.
```
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
**Example 4:** Saving the results in a pandas DataFrame and exporting it to a CSV file.
```

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
benchmark.get_result_df(round_num=3, return_ci=True).to_csv("result.csv")
```

**Appendix**

***How to build documentation?***
run ```bash build_docs.sh``` to generate markdown documentation for preview

***How to lint? Ruff !***
be sure you have installed pre-commit: ```pip install pre-commit```
now you can run ```pre-commit run``` to run pre-commit checks on changes
or ```pre-commit run --all-files``` to apply checks to all files
