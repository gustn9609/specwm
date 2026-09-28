# Speculative World Models — project page

Static project page for **Speculative World Models** (repo/URL slug: `specwm`).
No build step: `index.html` + `static/` + `assets/`. Served with GitHub Pages from the `main` branch root.

- Live URL: `https://gustn9609.github.io/specwm/`
- Preview locally: `python3 -m http.server 8000` then open `http://localhost:8000/`

## Main figure

`assets/figures/method_overview.svg` is hand-written SVG (editable in any text editor or Figma/Illustrator).
`assets/figures/method_overview.png` is a 2x raster export used for `og:image`.

Core idea shown in the figure: an **autoregressive** world model is the *draft* model (causal, cheap, proposes K frames),
and a **bidirectional** world model is the *target* model (full attention, verifies all K drafts in one pass).
Accepted prefix + first corrected frame are committed; loop repeats.

## Before publishing (TODO)

Search for `TODO` in `index.html`: authors/affiliations, abstract, results table, BibTeX, links, and the `noindex` meta tag.
