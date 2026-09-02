Here is the updated markdown table tracking the lifecycle of each rule change, including their inception and termination dates.
## Major Event Changes in MLB Data & Rules

## Impact on Model Stability

This table documents data infrastructure changes, equipment updates, and rule modifications that affect model performance and feature validity. Unlike rule changes that affect gameplay, data infrastructure shifts can silently corrupt features by changing measurement regimes or feature availability.

### COVID-Era Rules (2020-2022)

| Rule / Change | Inception Date | End Date | Status |
|---|---|---|---|
| "Ghost Runner" Rule (Runner on 2nd in extra innings) | July 2020 | Active | Permanent (Feb 2023) |
| 7-Inning Doubleheaders | August 2020 | October 2021 | Defunct |
| Taxi Squads (5-player travel reserve) | July 2020 | October 2021 | Defunct |
| Health Protocols (Kinexon tracking, PCR tests) | July 2020 | March 2022 | Defunct |
| Stadium Capacity Limits | April 2021 | July 2021 | Defunct |
| Blue Jays Displacement | July 2020 | July 30, 2021 | Defunct |
| Digital Infrastructure (Cashless, digital tickets) | April 2021 | Active | Permanent |

### Data Infrastructure, Equipment & Rule Changes (2015–2026)

| Event | Date | Type | Why It Matters for Your Models |
|---|---|---|---|
| Statcast sensor swap: TrackMan/radar → Hawk-Eye (12-camera optical) | Opening Day 2020 | Data Infrastructure | Spin rate/axis went from inferred to directly measured. Release point detection changed. Tracked-batted-ball coverage jumped. This is a **measurement regime change**, not a rule change — shifts raw feature distributions independent of field events. |
| Hawk-Eye camera frame-rate upgrade + bat tracking added | Mid-2023 | Data Infrastructure | High-speed cameras upgraded from 100 to 300 fps. Bat tracking (swing speed, swing path) began in H2 2023. New feature availability, not just new values. |
| Automatic intentional walk | 2017 | Rule | Eliminates 4 real pitches per IBB. Any feature depending on pitch-sequence counts, PA pitch totals, or pitcher fatigue proxies stops contributing data after this date. |
| Three-batter minimum for relievers | 2020 | Rule | Structural change to bullpen usage patterns. Breaks lookback features on reliever matchup/handedness optimization built on pre-2020 usage. |
| Juiced-ball era → ball deadened | Peak ~2017–19 → deadened 2021 | Equipment | MLB intentionally deadened the ball ahead of 2021 season after home run rates soared. Directly affects exit velo → HR probability mapping. |
| Humidor rollout expansion | 2020–2022 progression | Equipment | 2020: +3 parks (BOS/SEA/NYM) → 2021: 10 parks → 2022: all 30 parks. Ballpark-specific carry/HR factors from pre-2022 data invalid post-2022 for newly added parks. |
| Sticky-substance crackdown | June 2021 (initial), tightened Spring 2022 | Rule Enforcement | Pitchers threw 70% fewer high-spin elevated fastballs in 4 months after crackdown. **Hard level shift** at this boundary — any spin-rate-dependent stuff-quality feature has structural discontinuity here. |
| Universal DH made permanent | 2022 | Rule | NL pitcher-at-bat data disappears. Breaks any NL-specific run-scoring or lineup-construction features trained across this boundary. |
| Pitch clock (+ tightening 2024) | 2023, tightened 2024 | Rule | 20 sec with runners on → 18 sec in 2024. Clock restart timing changed. Affects pitcher rest/fatigue proxies and pickoff attempt features. |
| Shift ban (2 infielders per side of 2B, cleats in dirt) | 2023 | Rule | League-average shift usage was 33.6% in 2022 pre-ban. **Hard discontinuity** for BABIP, spray-angle, and positioning-dependent features. |
| Bigger bases (18"), pickoff/disengagement limits | 2023 | Rule | Base-stealing success rate and lead-distance features shift structurally. |
| Balanced schedule format | 2023 | Structural | Changes opponent-quality distribution per team-season. Affects any strength-of-schedule adjusted features. |
| ABS Challenge System (ball-strike review) | Full rollout 2026 | Rule / Data | Ball-strike calls not reviewable before 2026. New ground-truth zone data going forward — not comparable to pre-2026 umpire-only calls. Do not use post-2026 data to calibrate zone/count-value models on pre-2026 umpire training data. |
| "Golden At-Bat" proposal | Discussed, not adopted mid-2025 | Rule (Pending) | Would permit star slugger to hit out of order once per game in high-leverage situation. Watch-flag for lineup/count-value modeling if adopted. |

