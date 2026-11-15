# Pre-run hypotheses for the core comparison

- Depth (6 to 4 layers). Fewer blocks should lower wall-clock time per update. Reduced capacity may increase the final full-validation cross-entropy at the same token budget.
- Peak learning rate (0.0006 to 0.0003). Smaller updates may slow initial convergence and make validation loss change more smoothly. The minimum LR remains 0.00006.
- All comparisons use seed 1337, the same fixed training split, 2,500 updates, and 10,240,000 target tokens.
