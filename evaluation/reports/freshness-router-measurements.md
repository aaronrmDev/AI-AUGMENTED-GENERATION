# Freshness-Aware Data Router — Measurements

Simulated clock from 2026-01-01 (day 1) for 30 days. Every source is ingested hourly. Refresh runs at midnight, and review too in the freshness-aware arm. Every source is probed every 3 hours through Batch A's unrouted cascade (CAG -> MAG -> RAG, first hit wins), with tier thresholds carried over unchanged: CAG 0.43/0.35, MAG 0.42/0.37. Staleness is read from the assembled context using version-specific markers; no LLM runs. Schedules: prices change hourly; the return policy changes once, day 12 09:00; the catalog changes daily and hourly from day 10; the shipping guide never changes; the flash sale changes hourly until day 5 and then goes quiet; the owner's size preference changes day 15 09:00 and is probed by its owner and by another user of the same tenant. TTL part: warranty terms declared at 7 days, confirmed daily through day 7, changed directly in RAG on day 8. Question calibration: backpack-price: own 0.82, best other 0.22; return-policy: own 0.77, best other 0.26; blender-catalog: own 0.85, best other 0.15; shipping-guide: own 0.91, best other 0.32; flash-sale: own 0.78, best other 0.25; size-preference: own 0.52, best other 0.07; warranty-terms: own 0.80, best other 0.30. The change schedules are synthetic, one author wrote them and the rules, the cache is a CPU distilgpt2 proxy, and tier latency is cited from Batch A rather than re-measured here.

## Concept 9's freshness spectrum, routed

| Data type | Declared change | Scope | Source's paradigm | Route | Agrees |
|---|---|---|---|---|---|
| Stock prices, live scores | seconds | tenant | RAG | rag_only | yes |
| News, social media | minutes-hours | tenant | RAG | rag_only | yes |
| User session state | every turn | user | MAG | mag | yes |
| Product catalog | daily-weekly | tenant | CAG + RAG hybrid | cag_with_rag_backup | yes |
| Company policies, manuals | monthly-quarterly | tenant | CAG | cag_with_rag_backup | yes |
| Textbooks, reference | never | tenant | CAG | cag_with_rag_backup | yes |
| Code repositories | per commit | tenant | CAG + RAG | rag_only | no |
| User long-term preferences | gradually | user | MAG | mag | yes |

## Placement ablation over 30 simulated days

| Arm | Source | Probes | Stale only | Mixed | Current only | Neither | Stale rate | Served from CAG | Pre-loads | Never-served pre-loads | Foreign exposures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cache everything, batch refresh only | backpack-price | 241 | 210 | 0 | 31 | 0 | 87% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | return-policy | 241 | 5 | 0 | 236 | 0 | 2% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | blender-catalog | 241 | 147 | 0 | 94 | 0 | 61% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | shipping-guide | 241 | 0 | 0 | 241 | 0 | 0% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | flash-sale | 241 | 28 | 0 | 213 | 0 | 12% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | size-preference (owner) | 241 | 5 | 0 | 236 | 0 | 2% | 100% | 31 | 0 | 0 |
| cache everything, batch refresh only | size-preference (other user) | 241 | 5 | 0 | 236 | 0 | 2% | 100% | 31 | 0 | 241 |
| cache everything, invalidate on change | backpack-price | 241 | 0 | 0 | 241 | 0 | 0% | 13% | 31 | 0 | 0 |
| cache everything, invalidate on change | return-policy | 241 | 0 | 0 | 241 | 0 | 0% | 98% | 2 | 0 | 0 |
| cache everything, invalidate on change | blender-catalog | 241 | 0 | 0 | 241 | 0 | 0% | 39% | 31 | 0 | 0 |
| cache everything, invalidate on change | shipping-guide | 241 | 0 | 0 | 241 | 0 | 0% | 100% | 1 | 0 | 0 |
| cache everything, invalidate on change | flash-sale | 241 | 0 | 0 | 241 | 0 | 0% | 13% | 31 | 0 | 0 |
| cache everything, invalidate on change | size-preference (owner) | 241 | 0 | 0 | 241 | 0 | 0% | 98% | 2 | 0 | 0 |
| cache everything, invalidate on change | size-preference (other user) | 241 | 0 | 0 | 241 | 0 | 0% | 98% | 2 | 0 | 241 |
| RAG only | backpack-price | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | return-policy | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | blender-catalog | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | shipping-guide | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | flash-sale | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | size-preference (owner) | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| RAG only | size-preference (other user) | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 241 |
| freshness-aware | backpack-price | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| freshness-aware | return-policy | 241 | 0 | 0 | 241 | 0 | 0% | 98% | 2 | 0 | 0 |
| freshness-aware | blender-catalog | 241 | 0 | 0 | 241 | 0 | 0% | 30% | 10 | 0 | 0 |
| freshness-aware | shipping-guide | 241 | 0 | 0 | 241 | 0 | 0% | 100% | 1 | 0 | 0 |
| freshness-aware | flash-sale | 241 | 0 | 0 | 241 | 0 | 0% | 63% | 1 | 0 | 0 |
| freshness-aware | size-preference (owner) | 241 | 0 | 0 | 241 | 0 | 0% | 0% | 0 | 0 | 0 |
| freshness-aware | size-preference (other user) | 241 | 0 | 0 | 0 | 241 | 0% | 0% | 0 | 0 | 0 |

## Migration lag

| Source | Migration | Pattern shift | Migrated | Lag | Pre-loads before | Pre-loads after |
|---|---|---|---|---|---|---|
| blender-catalog | cag_with_rag_backup → rag_only | 2026-01-10 00:00 | 2026-01-11 00:00 | 24.0 h | 10 | 0 |
| flash-sale | rag_only → cag_with_rag_backup | 2026-01-05 00:00 | 2026-01-12 00:00 | 168.0 h | 0 | 1 |

## TTL bound on a silently changed cached source

| Arm | Source | Probes | Superseded text served from CAG |
|---|---|---|---|
| TTL factor 0.5 | warranty-terms | 241 | 20 |
| no TTL | warranty-terms | 241 | 185 |
