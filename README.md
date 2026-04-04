Unziip data.zip and extensive_features.zip. Run notebooks in the following order.

Order:
1. `new_features_extraction.ipynb` - creates 5 new features sets for every dataset. Code for features can be found in `src\features\new_features`
2. `xgboost.ipynb`, `logreg.ipynb`, `catboost.ipynb` - run experiments 
3. `analysis.ipynb` - collect results into tables

