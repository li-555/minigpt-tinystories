# Pre-run extension hypothesis

Replacing LayerNorm with RMSNorm removes mean-centering and the learned offset, while still normalizing the scale of each token's residual stream. With the same residual architecture, seed and token budget, RMSNorm may preserve similar validation loss, with a possible training-time improvement. GPU kernel dispatch and fusion can make measured runtime diverge from the arithmetic savings.

Comparison: the 6-layer baseline versus a single 6-layer RMSNorm run. Change all 13 normalization sites together and hold the data, initialization procedure, precision, context and optimizer fixed. Epsilon stays at 1e-5. Compare full public validation CE, perplexity and training time. A single paired seed is descriptive.
