# An Empirical Investigation of Contextualized Number Prediction
### by Spokoyny Daniel and Berg-Kirkpatrick Taylor

# Setup
- Data Tar: https://www.dropbox.com/s/lhi4i8kcqxlr7ou/3datasets.tar.gz?dl=0
- Move data folders to match this structure
    + models/
    + finetune_on_pregenerated.py
    + news/
    + news_dollar/
    + scidocs/
---
- poetry install
- poetry shell
- Uses [Wandb](http://wandb.ai/) for logging.

# Run
- To Run a sweep:
    - wandb sweep train_all.yml
    - wandb agent ##id_number##

# Cite
@inproceedings{berg-kirkpatrick-spokoyny-2020-empirical,
    title = "An Empirical Investigation of Contextualized Number Prediction",
    author = "Spokoyny, Daniel and Berg-Kirkpatrick, Taylor",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.385",
    doi = "10.18653/v1/2020.emnlp-main.385",
}
