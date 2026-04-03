# Baseline Integration Status

| Method | Strategy | Priority | Clone Needed | Current Status | Notes |
| --- | --- | --- | --- | --- | --- |
| CSRNet | Local implementation in `pack/models/` | High | No | Verified locally | Unified-protocol configs live under `pack/config/` |
| DM-Count | External repository adaptation | High | Yes | Active intake and pinned | Vendor snapshot goes in `external/baselines/dm_count/upstream/`; upstream `master@cc5f2132e0d1328909f31b6d665b8e0b15c30467` |
| CAN | Official external repository adaptation | High | Yes | Pinned and audited | Vendor snapshot lives in `external/baselines/can/upstream/`; upstream `master@d2e4d0425f578e556c1ab6017d326cff20466fad`; continue to local adaptation planning |
| SASNet | External repository adaptation | Medium | Maybe | Planned | Start after main baselines |
| CMTL | Investigate then decide | Medium-Low | Maybe | Planned | Older repository risk |
| M-SegNet | Citation or appendix only | Low | No | Deferred | Not a main counting baseline |

## DM-Count Pin Record

- Upstream URL: `https://github.com/cvlab-stonybrook/DM-Count`
- Upstream branch: `master`
- Pinned commit: `cc5f2132e0d1328909f31b6d665b8e0b15c30467`
- Pin date: `2026-04-04`
- Storage model: vendored snapshot under `external/baselines/dm_count/upstream/` with upstream `.git` removed
- Current stage: pinned, first-pass audit captured, data adaptation pending, metric/export pending

## CAN Pin Record

- Upstream URL: `https://github.com/weizheliu/Context-Aware-Crowd-Counting`
- Upstream branch: `master`
- Pinned commit: `d2e4d0425f578e556c1ab6017d326cff20466fad`
- Pin date: `2026-04-04`
- Storage model: vendored snapshot under `external/baselines/can/upstream/` with upstream `.git` removed
- Current stage: pinned, first-pass audit captured, local adaptation planning approved
