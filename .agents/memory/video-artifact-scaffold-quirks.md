---
name: Video (video-js) artifact traps
description: Two recurring traps when building a video artifact in this workspace — a misleading typecheck failure in the fresh scaffold, and browser autoplay policy silencing the music bed.
---

# Video artifact traps

## A fresh video-js scaffold fails typecheck for a misleading reason

New video artifacts inherit `lib: ["es2022"]` with no DOM, so the scaffold's own files report `Cannot find name 'window'` and bogus framer-motion `Variant` errors.

**Why:** the errors read like real bugs in freshly written scene code and send you chasing framer-motion typings that are fine.

**How to apply:** when a video artifact reports DOM globals missing, fix the artifact's tsconfig `lib` first (the web artifacts already include `dom`), then re-check — most of the error list disappears at once.

## Autoplay policy, not the wiring, is what silences the music bed

The prescribed audio pattern (`autoPlay` + `play().catch(() => {})`) leaves unmuted playback silently blocked in a normal browser tab, because the policy needs a user gesture.

**Why:** the failure is invisible — the video looks correct and plays through, it is just mute, so it survives every build and typecheck.

**How to apply:** treat a rejected `play()` as state, not a discarded error. Retry on the first document `pointerdown`/`keydown` and show a click-for-sound affordance in the preview only; keep the export/recording path free of the overlay, since headless capture runs with autoplay permitted.
