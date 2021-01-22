# Bookmark scorer

![build](https://github.com/tkngch/bookmark-scorer/workflows/build/badge.svg)

Bookmark relevance scoring methods. To be used to the bookmark manager ([Link](https://github.com/tkngch/bookmark-manager)).

## Implemented methods

- Averaging model
- Temporally-discounted averaging model
- Poisson model

## How to retrain a model and deploy

A model needs to be periodically retrained, to account for data-drift.

0. Install Python dependencies. Personally I use `venv` with `pip`.

```
python -m venv .venv
.venv/bin/pip -r ml_requirements.txt
```

1. Retrain the models by issuing the following command:

```
.venv/bin/python -m ml
```

The models are retrained on new data, and the best-performing model is compared
against the production model on their performance on the validation data-set. If
the retrained model outperforms the production model, the best retrained model
is serialised and stored as the new production model.

2. Commit the serialised model and push the commit.

Once the commit is merged to the main branch, Github action automatically
creates a new package with the retrained model.

3. Update the `bookmark-scorer` version on your application.
