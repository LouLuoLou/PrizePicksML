# Product Requirements Document (PRD)

**Project:** Multi-sport player prop Over/Under ML  
**Course context:** Spring 2026 student work (January–April 15, 2026)  
**Version:** 1.0  

## 1. Summary

Build a **line-relative binary classifier** that estimates **calibrated P(Over)** for a single player–stat market relative to a posted **line**, strictly using **information available before official match start** (per documented `feature_as_of` policy). Sports in scope: **basketball, baseball (MLB-style), soccer**. This repository is for **education and research reproducibility**, not for wagering or financial advice.

## 2. MVP scope

| Sport | MVP league (default) | Initial markets |
|-------|----------------------|-----------------|
| Basketball | NBA (configurable) | Points, minutes (separate models or features) |
| Baseball | MLB (configurable) | Hitter hits / pitcher strikeouts (pilot) |
| Soccer | Big-5 league (configurable) | Shots, shots on target (goals optional pilot) |

**Pilot honesty rule:** If data volume is insufficient for a sport, ship a **pilot** slice with clearly labeled reduced scope in the final report.

## 3. Users and outcomes

- **Primary user:** student / researcher running offline backtests and batch inference.
- **Primary output:** `P(Over)` after calibration, optional discrete **Over/Under** via threshold (default: `max(p, 1-p) >= 0.55`), optional **abstain** when below threshold.
- **Optional secondary:** point estimate of expected stat (regression) for sanity checks—not required for MVP pass/fail.

## 4. Label and push policy

- `y_over = 1` if `realized_stat > line`
- `y_over = 0` if `realized_stat < line`
- **Pushes** (`realized_stat == line`): **excluded** from training/evaluation by default; **push rate** must be reported.
- Settlement: assume **official league/boxscore corrections** as final; document any API-specific quirks in `DATA_DICTIONARY.md`.

## 5. Non-goals

- No profit guarantees, no bankroll optimization, no live betting integration.
- No scraping that violates terms of service (use licensed APIs or public datasets).
- Closing-line value (CLV) analysis **out of MVP** unless line timestamps are clean.

## 6. Ethics and compliance

- **Not wagering, financial, or health advice.** Users must comply with **age and jurisdictional** rules where they live.
- Model performance may show **no detectable edge**; that is an acceptable scientific outcome.
- Include limitations: late scratches, role uncertainty, efficient markets, label noise.

## 7. Success criteria (high level)

See `README.md` and plan acceptance criteria: walk-forward evaluation, leakage checklist, Brier/log loss/calibration, reproducible frozen build by **April 15**.

## 8. Semester notes (tone)

[Jan: locked MVP sports + public data stance] [Feb: push exclusion + no random CV for headline metrics] [Mar: opponent rolling shifted by one game] [Apr 15: tag + hash frozen artifact]
