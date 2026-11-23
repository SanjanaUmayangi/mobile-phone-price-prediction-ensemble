<!-- Copy this file to outputs/interpretation.md, replace everything in [brackets] with YOUR OWN findings
     (use the numbers from notebook 07). The dashboard shows interpretation.md on the "Model Insights" page. -->

**Best model.** On the untouched test set [model] performed best (MAE Rs. [x], RMSE Rs. [x], R² [x]). The gap to the
second-best model is [small/large] ([x]%).

**Why.** [Explain with evidence from your notebooks, for example: the forest-vs-number-of-trees curve (notebook 03) shows
averaging reduces variance; the boosting-rounds curve (04) shows ...; the error correlation between RF and XGBoost
was [x] (05), so stacking had [little/much] to gain; the ANN alone scored [x] and the best hybrid weight was [w] (06).]

**Stacking and hybrid.** [Did they beat the best single model? Why / why not? Think about data size, similar
base learners, correlated errors.]

**Over-fitting.** [Train-vs-test R² gap per model from `overfitting_gap.csv`.]

**Where the models fail.** [Errors by source (ikman vs Dialog), by condition, for expensive phones.]

**Limitations.** Prices are advertised/catalogue prices, not sale prices; two sources with different pricing
behaviour; RAM/storage extracted from ad text; few Dialog rows; snapshot in time.
