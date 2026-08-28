---
name: Expo/Metro dependency pin hazards
description: Why blanket pnpm overrides of Metro's transitive deps silently break the Expo web bundle, and how to scope them
---

# Blanket dependency overrides can break the Expo bundler

**Rule:** When pinning a transitive dependency workspace-wide (typically for a
vulnerability fix), check whether Metro depends on it. If Metro's declared range
does not include the pinned version, scope the override with a
`metro>{package}` entry pinned to the newest patched release *inside* Metro's
range, instead of loosening the global pin.

**Why:** A blanket pin can hand Metro a major version whose API it was never
written against. The failure surfaces only at bundle time, as an opaque runtime
`TypeError` deep inside a transform worker, with no mention of the override or
of dependency resolution. The Expo web app just serves a blank white page and a
500 for the bundle, so it reads as an app bug rather than a dependency problem.
Nothing in the app's own source has changed, and typecheck still passes, which
makes it easy to chase the wrong cause for a long time.

**How to apply:**
- Symptom to recognize: Expo web renders blank, workflow logs show
  `Web Bundling failed`, and the stack trace runs through a package Metro owns
  rather than through any app file.
- Confirm resolution rather than trusting the declared range: compare Metro's
  declared range in its own manifest against what the package manager actually
  linked for it.
- Prefer the newest patched release within the compatible major so the security
  fix is retained; only fall back to loosening the global pin if no such release
  exists.
- Leave a comment on the scoped override explaining the API incompatibility,
  otherwise a later dependency-hygiene pass will "clean it up" and reintroduce
  the breakage.
- Expo web bundling is not covered by typecheck, so verify a bundle actually
  builds after any dependency-hygiene change that touches the mobile artifact.
