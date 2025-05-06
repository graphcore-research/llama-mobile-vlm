# Block formats exploration code

Library code [`block_formats`](block_formats) and notebooks `*.pynb` + [`archive`](archive) to support our block formats investigation.

See [`Usage.ipynb`](Usage.ipynb) to get started.

Fisher sensitivity checkpoints are stored at `s3://graphcore-research/2025-04-block-formats/20250423-fisher`

> `aws s3 sync s3://graphcore-research/2025-04-block-formats/20250423-fisher/ out/20250423-fisher/`
> `aws s3 sync --dryrun out/PATH/ s3://graphcore-research/2025-04-block-formats/PATH/`


## Development

```sh
pytest block_formats/
```
