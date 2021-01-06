# Bookmark scorer

![build](https://github.com/tkngch/bookmark-scorer/workflows/build/badge.svg)

Bookmark relevance scoring methods. To be used to the bookmark manager ([Link](https://github.com/tkngch/bookmark-manager)).

## Implemented methods

- Average daily visits
- Poisson model

## How to retrain a model and deploy

A model needs to be periodically retrained, to account for data-drift.

0. Install Python dependencies. Personally I use `venv` with `pip`.

```
python -m venv .venv
.venv/bin/pip -r ml_requirements.txt
```

1. Retrain the Poisson model by issuing the following command:

```
.venv/bin/python -m ml
```

The model is retrained on new data, and its performance on validation dataset is
compared against that of the current production model. If the retrained model
outperforms the current production model, the retrained model is serialised and
automatically replaces the production model in `bookmark-scorer` library.

2. Commit the serialised model and push the commit.

Once the commit is merged to the main branch, Github action automatically
creates a new package with the retrained model.

3. Update the `bookmark-scorer` version on your application.
