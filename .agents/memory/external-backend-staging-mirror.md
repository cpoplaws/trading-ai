---
name: Testing a client against an external backend
description: Why the staging mirror for the external trading API lives in the workspace api-server instead of a local mock process
---

# Staging mirror for an external backend lives in an artifact service

**Rule:** When a frontend/mobile artifact talks to a backend that lives outside
the workspace, host the test double as routes on an existing workspace API
artifact, under a namespaced prefix, rather than as a standalone mock process on
a loose port. Namespace it so the client's own request paths stay byte-identical
underneath the prefix.

**Why:** The preview is served over https through a proxy, and only artifact
service ports are exposed through it. A mock listening on a bare container port
is reachable only over http, so browser mixed-content blocking kills every
request from the previewed app — the mock appears to work under curl and fails
only in the browser. Keeping the client's real paths intact matters just as
much: if the double is reachable at rewritten paths, the checks stop being able
to catch endpoint regressions, which is the main thing they exist for.

**How to apply:**
- Mount the mirror so `<prefix>` + the client's untouched path resolves, even
  when that means a doubled path segment. Do not edit the client to suit it.
- Gate it off in production behind an explicit env flag; leave it on in
  development so checks need no setup.
- Have it expose a health endpoint that asserts it is a demo/staging backend,
  and a reset endpoint that restores fixtures, so runs are deterministic.
- Make automated checks refuse to run unless the health endpoint confirms demo
  mode. That is the guardrail that stops a mistyped base URL from firing
  mutations at the real backend.
- Point the client at it with a client-specific env var that takes precedence
  over the shared one, so tests never disturb sibling artifacts' config.
