# Evaluation and data provenance

The checked-in CSVs supply the experiment inputs. The repository does not identify their publisher, collection procedure, or license. Establish provenance and reuse rights before redistributing the data.

The script computes target-derived geographic summaries before constructing validation folds. Those features can carry validation labels into the model inputs and make the printed cross-validation results optimistic. Some unsupervised preprocessing also uses both training and test features.

For a reliable estimate, reserve a time- or geography-based holdout first, fit preprocessing only on each training split, generate training target encodings out of fold, and compare against a simple historical-demand baseline. Evaluate the final blend on the untouched holdout.
