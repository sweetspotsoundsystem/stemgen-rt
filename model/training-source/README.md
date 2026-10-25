Matching StemgenRT 0.8.0 source distribution, including model definitions,
training, objectives, data/augmentation, Adam/EMA/RNG recovery, evaluation,
combined export and the exact independent partition exporter. Extract it and
read docs/training.md and docs/independent-stems.md.

The model weights are unchanged from 0.7.0. The archive excludes datasets,
checkpoints, papers and private experiment records. Its SHA-256 and exact
public source commit/tree are in sha256.json. CI verifies this archive against
the plugin version and checks that full training/export support is present.
