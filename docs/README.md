# NMFwisp Documentation Site

This directory contains a static documentation website for NMFwisp.

Preview from the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/docs/
```

You can also open `http://localhost:8000/`; the repository root now redirects
to the docs site.

If you start the server from inside this `docs/` directory instead, open:

```text
http://localhost:8000/
```

The site intentionally has no build step. It reuses the existing images in
`docs/example.png` and `docs/all_wisps.jpg`.
