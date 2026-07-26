---
name: Vercel Python+Next.js migration pattern
description: How to handle Vercel imports that have a Python backend alongside a Next.js frontend
---

# Vercel Python + Next.js Migration

**Rule:** When the imported Vercel project has a Python backend (FastAPI, Flask, etc.) and a Next.js frontend, port only the Next.js frontend to Vite+React. The Python backend is NOT ported into the Replit workspace — it remains external.

**Why:** The pnpm_workspace scaffold is Node.js-only. Python services can't run as workflow artifacts. The user's trading engine (Python) stays deployed on Railway/Render/etc., and the Vite frontend connects to it via env var.

**How to apply:**
- The Next.js frontend lives in `apps/dashboard/` or similar subdirectory — locate it first
- Convert `NEXT_PUBLIC_*` env vars to `VITE_*` in the frontend code
- Point `VITE_API_URL` at whatever URL the Python service is deployed at
- The workspace Express api-server scaffold is available for future Replit-native features, not for wrapping Python logic
- `VITE_WS_URL` for WebSocket connections from the frontend
- "Failed to fetch" errors in the dashboard are expected without the Python backend running
