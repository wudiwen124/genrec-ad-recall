# Configuration Notes

`paths.example.yaml` contains local/server path templates.

`experiments/` contains canonical, human-readable specifications for the final
model, SID generator, and evaluation. The current scripts expose the same
parameters as environment variables or CLI flags; they remain the executable
source of truth.

Do not commit machine-specific absolute paths, credentials, raw data, model
checkpoints, or generated logs.
