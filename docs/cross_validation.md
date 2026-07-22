# Cross-validation, model fitting, and nested OBD

This note explains the conceptual distinction underlying ROADMAP 4 Phase 4.

Almost—but the crucial distinction is that it does **not choose the test set by seeing which test set makes the model look best**.

Cross-validation creates several predetermined train/test partitions called folds. For five-fold CV:

1. Divide all patients into five representative groups.
2. Train on folds 2–5 and test on fold 1.
3. Train on folds 1 and 3–5 and test on fold 2.
4. Continue until every fold has served as the test set once.
5. Combine the five test results.

Thus it builds multiple models while systematically rotating which patients are held out. Every patient receives an out-of-sample prediction from a model that was not trained on that patient.

The nested OBD case adds another level:

```text
Outer training patients ── OBD search using its own validation split
                       └── selects the neural-network architecture

Outer test patients ───── untouched until that selected model is evaluated
```

For each outer fold, OBD uses only that fold’s training patients to choose the network architecture. The outer test fold remains completely hidden during model and architecture selection. This is what prevents optimistic performance estimates.

So there are two distinct workflows:

- **Logistic regression:** train the predetermined logistic model on each outer training fold and evaluate it on the corresponding test fold. No inner validation split is needed for architecture selection.
- **Neural network with OBD:** use an inner validation set to select/prune the architecture, then evaluate the resulting procedure on the untouched outer test fold.

Claude’s “CV of a procedure” wording is important. Nested CV estimates the performance of the complete procedure—“given new data, use OBD to select and fit a neural network”—rather than merely evaluating one architecture that was selected using the whole dataset.

For your SEER work, I agree with Claude’s proposed emphasis:

- Logistic cross-validation is clean, fast, and easy to explain as the primary regression analysis.
- Nested OBD cross-validation is the scientifically honest evaluation of the neural computational modeling pipeline.
- A single three-way train/validation/test split may still be useful during development and demonstration, but nested CV provides a stronger final performance estimate.

The concise answer is: **it selects test folds independently, then builds models appropriate to each fold. In the neural-network version, it also selects the architecture inside the training portion—never using the outer test patients for that choice.**
