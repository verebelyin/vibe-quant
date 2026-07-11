# Task: Reddit free-access research (beads ticket vibe-quant-7go7m.1)

Read AGENTS.md for project context. This is a WEB RESEARCH task — use web search extensively. Do NOT modify any code. Deliverable: `docs/reddit-access-research.md`. Do NOT commit.

## Context

This project scrapes r/algotrading via the public unauthenticated `https://www.reddit.com/r/<sub>/new.json` endpoint (see `vibe_quant/research/sources/reddit.py`: 6s inter-request floor, Retry-After handling, proper User-Agent per Reddit's guidance). It worked for months, then began hard-403ing from this IP on 2026-07-09 ("403 Blocked"). The user is open to registering an API token/app if needed.

## Questions to answer (cite sources with URLs + dates)

1. **Why does unauthenticated .json 403 now?** Current Reddit policy/enforcement on unauthenticated JSON access (datacenter vs residential IP treatment, UA policies, recent changes/announcements, community reports 2025-2026).
2. **Current official rate limits**: unauthenticated vs free OAuth (per Reddit's Data API terms + developer docs): requests/min, per-client vs per-IP, headers Reddit returns (X-Ratelimit-*).
3. **Free access paths, compared**: (a) fixing unauthenticated .json access (correct UA format, old.reddit.com, www vs oauth hosts — does anything reliably work in 2026?); (b) registered "script" app + OAuth2 (registration steps, what's free, quota, ToS constraints for research/personal use); (c) anything else free and ToS-clean (RSS feeds .rss endpoints! — check rate limits), embeds. Note: praw was deliberately removed from this project; raw httpx + OAuth is fine.
4. **Archive/backfill alternates** (secondary): current state of pullpush.io / arctic-shift / academic torrents for historical subreddit dumps — availability, terms, reliability in 2026.
5. **Recommendation**: one primary path + concrete scrape-cadence design around its rate limit (our workload: ~10 subs, listing pages + ~1 comments-fetch per post, one-time backfill of a few hundred posts/sub + periodic incremental).

## Deliverable format

`docs/reddit-access-research.md` — sections mirroring questions 1-5, every claim cited (URL + access date), recommendation section ends with a step-by-step "what to implement" list (env vars, endpoints, headers, backoff) and a HITL checklist of anything only a human can do (e.g. create Reddit app, get client_id/secret).

Also write a 5-line executive summary at the top.
