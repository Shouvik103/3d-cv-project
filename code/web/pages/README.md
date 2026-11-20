# Reconstruction Lab on GitHub Pages

The read-only edition uses the workbench's existing viewer and benchmark UI. The
published snapshot contains browser geometry, GLB downloads, mesh previews,
and selected measurements. Each build also reads the current two-method benchmark
collection from `validation/experiments_*` using the same catalog as the workbench. It leaves out original uploads,
the workbench database and job records, raw logs, and intermediate caches.

Build and preview the checked-in snapshot from the repository root:

```sh
python3 code/web/pages/build_gallery.py build dist/gallery
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist/gallery
```

To refresh the snapshot locally, run `build_gallery.py snapshot` with the project's
Python environment and a fresh output directory. It reads completed runs from
`outputs/web_ui`. Review the generated assets
before replacing `code/web/pages/snapshot`. The Pages workflow publishes the
bundle on `main` after the repository's Pages source is set to GitHub Actions.
