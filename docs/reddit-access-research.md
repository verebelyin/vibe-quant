# Reddit free-access research after the 2026 anonymous-JSON shutdown

## Executive summary

1. Reddit announced on 2026-05-28 that it was shutting down unauthenticated `.json`; the current API wiki says traffic without OAuth or login credentials is blocked, so the 2026-07-09 failure is consistent with policy enforcement, not an exhausted anonymous quota ([Reddit announcement, 2026-05-28](https://www.reddit.com/r/modnews/comments/1tq9vxo/protecting_communities_from_scrapers_and_platform/); [Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).
2. There is no current anonymous free tier; an eligible, approved OAuth client receives 100 queries/minute per client ID, averaged over ten minutes, and should monitor Reddit's three `X-Ratelimit-*` headers ([Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).
3. The primary path is prior approval plus a registered `script` client using raw OAuth2 and `oauth.reddit.com`; approval, not code, is the first blocker because Reddit ended self-service API access for new tokens in 2025 ([approval announcement, 2025](https://www.reddit.com/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/); [Responsible Builder Policy, updated 2026-06-05](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy); accessed 2026-07-11).
4. RSS is at best a degraded listings fallback: Reddit identified it as another scraping surface, and June 2026 users observed roughly one anonymous request/minute plus 429/403 failures; embeds cannot discover or ingest posts/comments ([Reddit announcement, 2026-05-28](https://www.reddit.com/r/modnews/comments/1tq9vxo/protecting_communities_from_scrapers_and_platform/); [RSS report, 2026-06-12](https://www.reddit.com/r/rss/comments/1u3qqmk/did_reddit_change_the_rate_limit/); [Embeds Terms](https://redditinc.com/policies/embeds-terms); accessed 2026-07-11).
5. PullPush, Arctic Shift, and torrents remain useful for bounded historical recovery, but they are unofficial, lagged/best-effort, and not a ToS-clean substitute for Reddit approval or deletion compliance ([PullPush](https://pullpush.io/); [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift); [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy); accessed 2026-07-11).

Research cutoff and access date for every source below: **2026-07-11**. “Official” means Reddit-authored policy, help, documentation, or an administrator announcement. Community reports are operational evidence, not guarantees.

## 1. Why unauthenticated `.json` returns 403 now

### Official change

On 2026-05-28 Reddit explicitly announced that it would shut down unauthenticated `.json` endpoints because they enabled scraping “without accountability”; logged-in and authenticated access would remain. The same announcement described large-scale scraping, spam networks, agentic account creation, and automated abuse as the reasons for tightening automated access ([Reddit r/modnews announcement, 2026-05-28](https://www.reddit.com/r/modnews/comments/1tq9vxo/protecting_communities_from_scrapers_and_platform/); accessed 2026-07-11).

The current Data API Wiki removes any ambiguity: clients must authenticate with a registered OAuth token, Reddit may throttle or block unidentified clients, and traffic without OAuth or login credentials “will be blocked” rather than receive the default rate limit. A correct User-Agent and a six-second request floor therefore cannot make anonymous JSON supported again ([Reddit Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).

This was a real ecosystem-wide transition, not unique to vibe-quant. Developers reported anonymous JSON 403s on 2026-05-29 and widespread failures in self-hosted clients on 2026-05-30, immediately after the announcement ([r/redditdev report, 2026-05-29](https://www.reddit.com/r/redditdev/comments/1tr0fw8/is_anynomous_access_to_the_json_endoints_gone/); [r/selfhosted report, 2026-05-30](https://www.reddit.com/r/selfhosted/comments/1trz8ld/did_reddit_kill_unauthenticated_json_requests/); accessed 2026-07-11).

### Datacenter versus residential IPs

Reddit separately documents that an IP in a hosted-service-provider netblock requires a valid OAuth token or a logged-in session. That explains why cloud/datacenter traffic was blocked earlier and more consistently than home traffic ([Developer Platform & Accessing Reddit Data, updated 2026-05-28](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data); accessed 2026-07-11). A March 2026 report matched the rule: the same public JSON call worked locally but 403ed from Google Cloud; a 2025 report captured Reddit's network-policy block page from an AWS address ([Google Cloud report, 2026-03-20](https://www.reddit.com/r/redditdev/comments/1ryzhyc/accessing_reddit_json_payload/); [AWS report, 2025-01-09](https://www.reddit.com/r/redditdev/comments/1hxgmpu/); accessed 2026-07-11).

After the May shutdown, a residential success is not a supported anonymous tier. It may be staged enforcement, IP reputation, or a browser silently supplying a logged-in credential; Reddit's published rule remains that unauthenticated traffic is blocked ([Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); [browser-context discussion, 2026-05-29](https://www.reddit.com/r/redditdev/comments/1tr0fw8/is_anynomous_access_to_the_json_endoints_gone/); accessed 2026-07-11).

### User-Agent and the July 9 onset

Reddit still requires a truthful, unique User-Agent in the form `<platform>:<app ID>:<version> (by /u/<username>)`; generic library UAs are drastically limited. This is necessary for approved OAuth use but no longer sufficient for anonymous JSON ([Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).

No official source found announces a global change specifically on 2026-07-09. The supportable inference is that the already-announced shutdown reached this egress IP through staged enforcement, reclassification, or loss of prior tolerance. Browser-UA spoofing, TLS-fingerprint tricks, residential proxies, copied cookies, and host rotation should not be used: Reddit prohibits masking identity and circumventing access controls ([Data API Terms §§2.4, 3.2](https://redditinc.com/policies/data-api-terms); [Developer Terms §§4.2, 6](https://redditinc.com/policies/developer-terms); accessed 2026-07-11).

## 2. Current official rate limits

| Access mode | Current quota | Bucket/window | Returned headers |
|---|---:|---|---|
| Unauthenticated `.json` | **Blocked; no current anonymous quota** | Not applicable. The former allowance was 10 QPM, but the current wiki says unauthenticated traffic is blocked and the default limit does not apply. | Do not depend on rate headers on a network-policy 403. |
| Eligible free OAuth Data API client | **100 QPM per OAuth client ID** | Averaged over a rolling/current ten-minute window, allowing bursts—approximately 1,000 requests per ten minutes, but 100 QPM is the normative limit. | `X-Ratelimit-Used`, `X-Ratelimit-Remaining`, and `X-Ratelimit-Reset`. |

Sources for the current rows and headers: [Reddit Data API Wiki, updated 2026-05-11](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki) (accessed 2026-07-11). Reddit's June 2023 announcement documents the superseded 10-QPM non-OAuth allowance and the then-new 100-QPM OAuth allowance, confirming that 10 QPM is historical rather than the 2026 rule ([Reddit API facts, 2023-06-30](https://redditinc.com/news/apifacts); accessed 2026-07-11).

The current bucket is explicitly **per client ID**, not per IP and not per short-lived access token. Reddit retains discretion to alter limits, charge for future use, require a separate agreement for commercial access or excess research, suspend access, and block circumvention ([Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); [Data API Terms §§2.9, 3.1–3.2, 5.2](https://redditinc.com/policies/data-api-terms); accessed 2026-07-11).

## 3. Free access paths compared

### A. Repairing anonymous JSON

| Attempt | 2026 finding | Verdict |
|---|---|---|
| Better UA | Required for identified clients, but does not replace OAuth. | Necessary after approval; not an anonymous fix. |
| `old.reddit.com` | A presentation host, not an authentication mechanism; the shutdown applies to unauthenticated `.json` endpoints. | Not reliable or supported. |
| `www.reddit.com` versus `oauth.reddit.com` | Token acquisition uses `www`; bearer API calls belong on `oauth.reddit.com`. Calling the OAuth host without a bearer token is not a workaround. | Use the documented split only with approved credentials. |
| Residential proxy, browser UA/cookies, TLS fingerprint | Attempts to appear logged-in or evade network policy are brittle and may mask identity/circumvent controls. | Reject. |

Sources: [shutdown announcement, 2026-05-28](https://www.reddit.com/r/modnews/comments/1tq9vxo/protecting_communities_from_scrapers_and_platform/), [OAuth2 guide](https://github.com/reddit-archive/reddit/wiki/OAuth2), [Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki), and [Developer Terms](https://redditinc.com/policies/developer-terms) (accessed 2026-07-11).

**Conclusion:** there is no reliable, free, ToS-clean anonymous JSON path in 2026.

### B. Approved `script` app plus OAuth2

This is the technical fit for the existing raw-`httpx` collector. A `script` app is a confidential client running on hardware the developer controls. For public read-only endpoints it can request an application-only bearer token by sending HTTP Basic `(client_id, client_secret)` and `grant_type=client_credentials` to `POST https://www.reddit.com/api/v1/access_token`. Data calls then use `https://oauth.reddit.com` with `Authorization: bearer <token>`. Application-only tokens expire and do not return refresh tokens, so the client obtains a new token when needed ([OAuth2 “Application Only OAuth” guide](https://github.com/reddit-archive/reddit/wiki/OAuth2); accessed 2026-07-11).

The older script quick start demonstrates `grant_type=password` with the developer account, but that is unnecessary credential exposure for public read-only collection unless Reddit's approval or required endpoint specifically demands account context ([OAuth2 script quick start](https://github.com/reddit-archive/reddit/wiki/oauth2-quick-start-example); accessed 2026-07-11).

The important 2026 change is administrative: in 2025 Reddit ended self-service access for new OAuth tokens and required approval; the current Responsible Builder Policy requires explicit approval before API access. Developers are directed to Devvit first and to a request form when the use case is unsupported there ([Reddit approval-process announcement, 2025](https://www.reddit.com/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/); [Responsible Builder Policy, updated 2026-06-05](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy); [Data API request form](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164); accessed 2026-07-11).

Free access is conditional on approval and eligibility. Reddit requires permission/contract for commercial uses, including use by a business or as part of a monetized product; its current policy says research must use the RFR program and prohibits Reddit data as model-training input without explicit permission. The application must truthfully describe vibe-quant's trading-discussion analysis rather than selecting a more favorable but inaccurate category ([Developer interfaces, updated 2026-05-28](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data); [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy); [Developer Terms](https://redditinc.com/policies/developer-terms); accessed 2026-07-11).

Community reports from late 2025 through 2026 describe redirects away from `/prefs/apps`, CAPTCHA loops, delayed responses, and rejections. These do not establish an approval probability, but they make approval a real project dependency rather than a same-day registration assumption ([approval announcement discussion](https://www.reddit.com/r/redditdev/comments/1oug31u/introducing_the_responsible_builder_policy_new/); [2026 access discussion](https://www.reddit.com/r/redditdev/comments/1sy1y8g/has_any_developer_here_got_access_to_reddit_api/); accessed 2026-07-11).

### C. Other free, cleaner paths

**RSS.** Subreddit feeds such as `https://www.reddit.com/r/<sub>/new/.rss` remained reachable for some users, but Reddit's shutdown announcement called RSS another common scraper surface and made no availability or rate promise. In June 2026 multiple users observed one successful anonymous request followed by `X-Ratelimit-Remaining: 0` and about a 60-second reset, plus 429s and some persistent 403s ([Reddit announcement, 2026-05-28](https://www.reddit.com/r/modnews/comments/1tq9vxo/protecting_communities_from_scrapers_and_platform/); [RSS reports, 2026-06-12](https://www.reddit.com/r/rss/comments/1u3qqmk/did_reddit_change_the_rate_limit/), [2026-06-16](https://www.reddit.com/r/rss/comments/1u7dmk5/reddit_rate_limiting_workaround_stopped_working/); accessed 2026-07-11). RSS is listings-only for this design, carries fewer fields, has no documented stable quota, and cannot replace one comments request per post. Use only as an explicitly degraded, low-frequency emergency mode—not as a way around rejected API approval.

**Embeds/oEmbed.** Reddit says reasonable embed use is generally free, but the Embeds Terms limit it to making a known post or comment visible. It offers neither subreddit discovery nor bulk backfill and cannot be repurposed as an ingestion API ([Developer interfaces, updated 2026-05-28](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data); [Embeds Terms](https://redditinc.com/policies/embeds-terms); accessed 2026-07-11).

**Reddit for Researchers (RFR).** Approved non-commercial academics receive free BigQuery access to a monthly updated dataset covering a rolling five years with a six-month delay. Eligibility requires accredited-university affiliation, an institutional email and sponsor, and ethics approval/exemption; access is project-specific and limited to one year. It is not a live collector backend and is not appropriate unless this project genuinely qualifies ([RFR Program, updated 2026-06-02](https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program); accessed 2026-07-11).

## 4. Archive/backfill alternatives

### PullPush

PullPush exposes Pushshift-compatible JSON search endpoints for submissions and comments, including `subreddit`, `after`, `before`, and result sizes up to 100. Its documentation says it relies on Reddit BitTorrent data compiled by third parties and that removal requests must be filed separately because the source torrents cannot be changed ([PullPush API documentation](https://pullpush.io/); accessed 2026-07-11).

The service publishes no SLA or completeness/freshness guarantee. Its terms provide the service “as is” and contain unusual arbitration/jurisdiction language, which is an additional operational/legal warning ([PullPush Terms](https://www.pullpush.io/tos.html); accessed 2026-07-11). It is convenient for a small experiment, but not suitable as the authoritative live source.

### Project Arctic Shift

Arctic Shift offers bulk dumps, a downloader for users/smaller subreddits, a web search UI, and an API the maintainer explicitly calls “limited.” At the cutoff its repository showed 37 releases and a May 2026 release published on 2026-06-11, demonstrating recent activity but not an SLA ([Arctic Shift repository](https://github.com/ArthurHeitmann/arctic_shift); accessed 2026-07-11).

It is the strongest technical third-party candidate for a bounded subreddit backfill, but its data has research-relevant caveats: community comparison identifies gaps around April–June 2023, score-refresh differences, and different deleted-author behavior, while the project documents transformations and separate removal handling ([Arctic Shift data notes](https://github.com/ArthurHeitmann/arctic_shift/blob/master/file_content_explanations.md); [coverage comparison, 2025-05-18](https://www.reddit.com/r/pushshift/comments/1kpaoj1/how_comprehensive_are_the_torrent_dumps_after_2023/); accessed 2026-07-11).

### Academic Torrents / subreddit dumps

A 2026 community release provides separate comment and submission files for the top 40,000 subreddits through December 2025, combining pre-April-2023 Pushshift data with later Arctic Shift collection. Files are zstd-compressed NDJSON and can be selected by subreddit, avoiding a multi-terabyte full download ([release and format notes, 2026](https://www.reddit.com/r/pushshift/comments/1r5z42j/separate_dump_files_for_the_top_40k_subreddits/); accessed 2026-07-11).

These dumps are peer/storage dependent, stale for incremental use, and immutable: deletions cannot propagate into an already-published torrent. The publisher warned that the 3.2-TB release needed to be redownloaded/reseeded and might take time to gain availability ([same release, “Seeding”](https://www.reddit.com/r/pushshift/comments/1r5z42j/separate_dump_files_for_the_top_40k_subreddits/); accessed 2026-07-11).

### Policy conclusion for archives

Third-party availability does not grant permission from Reddit or content rightsholders. Reddit's current policy says research outside RFR violates the policy, requires deletion handling, and restricts commercial/data-mining/model uses. Archives should therefore be disabled unless a human makes and records a separate policy/legal decision; they should never be a silent failover from a rejected OAuth application ([Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy); [Data API deletion guidance](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); [Developer Terms](https://redditinc.com/policies/developer-terms); accessed 2026-07-11).

## 5. Recommendation and scrape cadence

### Primary recommendation

Apply truthfully for Data API access, obtain approval, register/receive a confidential `script` client, and migrate the collector to application-only OAuth with raw `httpx`. This is the only identified path that provides official structured listings and comments, works from hosted networks, and has a documented free quota ([Developer interfaces](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data); [Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).

The recommendation is conditional. If Reddit does not approve the disclosed trading-research use, there is no equivalent free, ToS-clean JSON ingestion route. Pause full Reddit ingestion; at most, enable sparse RSS listings while requesting written clarification. Do not evade the block.

### Cadence design around 100 QPM

Use one process-wide limiter per OAuth client ID, initially capped at **60 requests/minute sustained** (one request/second), leaving 40% headroom. Treat returned `X-Ratelimit-Remaining` and `X-Ratelimit-Reset` as authoritative because Reddit reserves discretion over enforcement ([Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); [Data API Terms](https://redditinc.com/policies/data-api-terms); accessed 2026-07-11).

- **One-time backfill:** fetch `limit=100` listing pages using the `after` cursor, one subreddit at a time, stopping near 300 posts/subreddit. Ten subreddits require about 30 listing calls. With one comments call for each of 3,000 posts, the conservative upper bound is 3,030 requests—about **51 minutes at 60 QPM**. Posts with zero comments reduce the actual total. Reddit listings use `after`/`before` cursors and allow up to 100 items per request ([Reddit API listing documentation](https://www.reddit.com/dev/api/); accessed 2026-07-11).
- **Incremental:** every **15 minutes**, fetch one `/new` page for each of ten subreddits (10 baseline calls), stop at a known ID/time boundary, and fetch comments only for unseen posts with `num_comments > 0`.
- **Comment freshness:** retain one top-comments fetch at ingestion. If later freshness is required, permit one bounded delayed re-fetch for recent posts; never sweep every historical thread.
- **Header reserve:** when remaining quota approaches a small reserve (for example, 10), wait until reset rather than issuing the request.
- **Errors:** on 429, honor `Retry-After`; otherwise use capped exponential backoff with jitter for 429/5xx/network errors. On 401, renew the token once. On persistent 403, stop and surface an approval/authorization error rather than retrying or rotating identity.

### What to implement

1. Add secret env vars `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`; add non-secret `REDDIT_USER_AGENT="python:vibe-quant:<version> (by /u/<approved-account>)"`. Do not require or store the Reddit account password for the app-only read path. Keep credentials out of code, logs, fixtures, and the state DB ([OAuth2 application-only guide](https://github.com/reddit-archive/reddit/wiki/OAuth2); [Developer Terms](https://redditinc.com/policies/developer-terms); accessed 2026-07-11).
2. Acquire a token with `POST https://www.reddit.com/api/v1/access_token`, HTTP Basic auth `(client_id, client_secret)`, form field `grant_type=client_credentials`, and the truthful UA. Cache it only in memory and renew based on `expires_in`; app-only tokens have no refresh token ([OAuth2 application-only guide](https://github.com/reddit-archive/reddit/wiki/OAuth2); accessed 2026-07-11).
3. Call `https://oauth.reddit.com/r/{subreddit}/new` and `https://oauth.reddit.com{permalink}` with `Authorization: bearer <token>`, the same UA, `Accept: application/json`, and `raw_json=1` ([OAuth2 guide](https://github.com/reddit-archive/reddit/wiki/OAuth2); accessed 2026-07-11).
4. Replace the per-instance anonymous six-second floor with one global 60-QPM limiter shared by listings and comments; dynamically defer to `X-Ratelimit-Used`, `X-Ratelimit-Remaining`, and `X-Ratelimit-Reset` ([Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).
5. Preserve `Retry-After`; add jittered capped exponential backoff for 429/transient failures, a single token renewal on 401, and a fail-fast actionable error for persistent 403.
6. Backfill sequentially with `limit=100` and `after`; checkpoint every page, deduplicate by Reddit fullname/ID, and resume safely. Run incrementals every 15 minutes and request comments only for unseen posts.
7. Add deletion compliance: revalidate/expire stored Reddit content, delete removed posts/comments and identifying author fields, and document retention. Reddit recommends routinely deleting stored user data/content within 48 hours as the safest compliance pattern ([Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki); accessed 2026-07-11).
8. Keep RSS behind a degraded feature flag, globally no faster than one request/minute and listings-only. Never use copied cookies, proxy rotation, UA spoofing, or third-party archives as automatic failover.

### HITL checklist

- [ ] A human decides and documents whether the use is personal/non-commercial, commercial, or genuinely academic; do not submit a misleading category.
- [ ] A human reads and accepts the current [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy), [Developer Terms](https://redditinc.com/policies/developer-terms), and [Data API Terms](https://redditinc.com/policies/data-api-terms) (accessed 2026-07-11), including retention/deletion and AI/model restrictions.
- [ ] A human with a Reddit account in good standing submits the [Data API access request](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164) with exact subreddits, fields, storage/retention, volume estimate, security controls, and the trading-discussion-analysis purpose.
- [ ] After written approval, a human creates or receives the `script` app/client, records the client ID/secret once, and supplies both through the deployment secret store.
- [ ] A human confirms that the approval covers analysis and storage for vibe-quant—not merely display, moderation, or academic research—and asks Reddit for clarification if ambiguous.
- [ ] A human makes a separate policy/legal decision before enabling PullPush, Arctic Shift, or torrent backfill; without that decision, all three remain disabled.
