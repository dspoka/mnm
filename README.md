# mnm
- An Empirical Investigation of Contextualized Number Prediction

# Setup
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