# Llama-Mobile slides

The PDF exporter uses Python 3, Node.js 22 or newer, and `google-chrome`.

Before presenting or exporting, copy `Graphik-Regular-Web.woff2` and
`Graphik-Medium-Web.woff2` into `assets/fonts/` (create the directory if needed).
Font files are ignored by Git; without them, the slides use fallback fonts.

## Present

Present:

```sh
python3 tools/serve.py
```

Prepare `out/{date}-llama-mobile.pdf`:

```sh
node tools/export-pdf.mjs
```

Regenerate figures:

```sh
python3 tools/plot_efficient_formats.py  # for assets/img/efficient-formats.svg
python3 tools/plot_ste.py  # for assets/img/ste.svg
```

Note that `plot_efficient_formats.py` caches centroids in `assets/data/efficient-formats.json`. Increment `VERSION` in the script to invalidate the cache and refit centroids (slow).
