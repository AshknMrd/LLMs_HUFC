# DRAGON Preprocessing
This repository contains the preprocessing scripts for the [DRAGON challenge](https://dragon.grand-challenge.org/).

If you are using DRAGON resources, please cite the following article:

> J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)

Download the citation file for your reference manager: [BibTeX](https://github.com/DIAGNijmegen/dragon/blob/main/citation.bib) | [RIS](https://github.com/DIAGNijmegen/dragon/blob/main/citation.ris)


## Installation
`dragon_prep` can be pip-installed:

```
pip install dragon_prep
```

Alternatively, it can be installed from source:

```
git clone https://github.com/DIAGNijmegen/dragon_prep
cd dragon_prep
pip install -e .
```

The Docker can be built after cloning the repository. The anonymisation code is not included due to privacy concerns, so you have to uncomment [copying and installing the diag-radiology-report-anonymizer](https://github.com/DIAGNijmegen/dragon_prep/blob/data-preparation/Dockerfile#L20-L21). The unmodified version is included to reflect the exact code used to prepare the DRAGON challenge resources.

```
git clone https://github.com/DIAGNijmegen/dragon_prep
cd dragon_prep
nano Dockerfile  # uncomment copying and installing the diag-radiology-report-anonymizer
./build.sh
```

If ran successfully, this results in the Docker container named `dragon_prep:latest`.

## Resources
The preprocessing scripts for the synthetic datasets can be found in [`src/dragon_prep`](/src/dragon_prep) and are the script called `Task1xx_Example_yy.py`. The preprocessing scripts for the datasets used in the test leaderboard for the DRAGON challenge can be found in [`src/dragon_prep`](/src/dragon_prep) and are the script called `Task0xx_yy.py`. The datasets for the validation leaderboard are derived from the development data, using the [`src/dragon_prep/make_debug_splits.py`](src/dragon_prep/make_debug_splits.py) script. For the DRAGON challenge, all datasets were preprocessed using the [`preprocess.sh`](preprocess.sh) script.

## Usage
The synthetic datasets can be generated with any number of samples.

After installing the `dragon_prep` module:

```python
python src/dragon_prep/Task101_Example_sl_bin_clf.py \
    --output_dir=./output \
    --num_examples={set any number you like}
```

Or, using the Docker container:

```bash
docker run --rm -it \
    -v /path/to/store/data:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task101_Example_sl_bin_clf.py \
        --num_examples={set any number you like}


# ... same for Task102_Example_sl_mc_clf.py, Task104_Example_ml_bin_clf.py, Task105_Example_ml_mc_clf.py, Task106_Example_sl_reg.py, Task107_Example_ml_reg.py, Task108_Example_sl_ner.py, Task109_Example_ml_ner.py
# for Task103_Example_mednli.py, setting the number of examples is not supported
```

The preprocessing scripts for the tasks in the DRAGON benchmark are included for transparancy and to provide building blocks to process your own data. To run the end-to-end script using your own data, you can turn off the anonymisation functionality:

```python
prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name, tag_phi=False, apply_hips=False)
```

## Managed By
Diagnostic Image Analysis Group, Radboud University Medical Center, Nijmegen, The Netherlands

## Contact Information
Joeran Bosma: Joeran.Bosma@radboudumc.nl
