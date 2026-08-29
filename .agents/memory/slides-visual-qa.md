---
name: Visual QA for slide decks
description: How to inspect every slide of a deck at once without being misled by viewport-relative units.
---

# Visual QA for slide decks

Do not judge slide layout from a tall-viewport screenshot of the all-slides route. Slide components position everything in `vw`/`vh`, and `vh` resolves against the *browser viewport*, not the 16:9 slide box — so a 2400px-tall capture squashes every absolutely-positioned element and invents overlaps that do not exist in the real deck.

Instead, export the deck to PDF and rasterize the pages into one contact sheet:
`pdftoppm -jpeg -r 40 <deck>.pdf p` then append the pages into a grid image and read that.

**Why:** the export renders each slide at true 16:9, so what the sheet shows is what the audience gets, and one image covers the whole deck instead of one screenshot per slide.

**How to apply:** after writing or restyling slides, run the export once, QA the contact sheet, fix, re-export. Genuine overflow shows up as text colliding with the bottom rail or the page-number lockup — the most common cause is a two-line wrap in a row whose sibling rows are single-line.
