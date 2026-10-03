# Deploying PRISE

> **Note for readers of the public repo:** this describes the author's own
> deployment, which runs from a private repository that carries the derived
> `data/processed` artifacts. Those artifacts are not redistributed here —
> see README, "You will need the artifacts".

Stdlib `http.server` backend + vanilla-JS SPA, plus numpy/scipy for the engine.
It loads `data/processed` into memory at startup.

## Footprint (measured, not estimated)

| | |
|---|---|
| Payload | code ~4 MB + `data/processed` ~125 MB (440 files) |
| Peak RSS | ~376 MB after warming every endpoint |
| Deps | numpy, scipy (the `web/` backend itself is stdlib-only) |
| Endpoint latency | all major endpoints < 30 ms, measured locally |

## Target: Render free tier

512 MB RAM, 0.1 CPU, no credit card. `render.yaml` is a Blueprint, so the
service configures itself and every push to `master` redeploys.

**Deliberately not Git LFS.** Render does not fetch LFS objects at build time,
so `data/processed` is committed as ordinary git blobs. The largest single file
is 17.7 MB — well inside GitHub's limits — and the repo lands around 125 MB.

### One-time setup

1. Create an empty repo at <https://github.com/new> — name `PRISE`, no README,
   no .gitignore, no license (this repo already has them).
2. Push:

   ```bash
   cd C:/Users/yassi/PRISE
   git remote add origin https://github.com/<user>/PRISE.git
   git push -u origin master
   ```

3. Go to <https://dashboard.render.com/blueprints>, connect the GitHub repo, and
   Render reads `render.yaml`. No manual service configuration.

### Thereafter — push to deploy

```bash
git push origin master     # Render rebuilds and goes live automatically
```

## Known risks on the free tier

- **Memory is tight.** 376 MB peak against a 512 MB ceiling leaves ~136 MB of
  headroom, and that peak was measured on Windows — Linux will differ. If the
  service OOMs, the fix is to lazy-load the three largest per-series artifacts
  (`metadata_quality.jsonl` 17.7 MB, `boed_recommendations.jsonl` 17.1 MB,
  `information_content.jsonl` 13.3 MB) instead of reading them all at startup.
- **Startup is slow.** Parsing 125 MB of JSON on 0.1 CPU takes far longer than
  the few seconds it takes locally. Render's health check is `/api/health`,
  which only answers once `STORE.load()` finishes.
- **The service sleeps** after 15 minutes idle; the next visitor waits ~1 minute
  for it to wake. This is inherent to the free plan.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `PRISE_HOST` | `127.0.0.1` | bind address; deployment sets `0.0.0.0` |
| `PRISE_PORT` / `PORT` | `8000` | listen port; Render injects `PORT` |

Local behaviour is unchanged: `python web/server.py` still binds loopback only.
