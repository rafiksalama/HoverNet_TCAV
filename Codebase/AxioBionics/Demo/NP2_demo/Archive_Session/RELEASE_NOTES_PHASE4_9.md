# NP2 Demo – Phases 4‑9 Enhancements

## Phase 4 – Power & Settings
- Added a reusable Settings modal with toggles for sound, demo mode, display preference, and global power limits.
- Clinician Mode is now PIN-protected; advanced actions (template editing, EMG triggers) require the PIN.
- Implemented user login/logout flow with subscription tiers that gate template sharing and analytics features.

## Phase 5 – Controls & Audio Feedback
- Channel adjustments now flash and show the updated percentage, with higher/lower pitch tones for increment/decrement (configurable via Settings).
- Channel metric display switches between Voltage, Current, Charge per Pulse, or Intensity based on user preference.

## Phase 6 – EMG / EMS Insights
- Introduced a dedicated `EMG / Analytics` screen showing baseline capture, monthly peaks, monitoring logs, triggers, and curated patient journeys (plan-gated).
- Simulated EMG data feeds the monitoring log, and EMG-triggered EMS can be toggled in clinician mode.

## Phase 7 – Accounts & Subscription Gating
- Added a login screen where users select their role and plan; gating logic ensures Pro/Enterprise plans unlock advanced features.
- Template export/import and creation are now plan-aware with inline upgrade messaging.

## Phase 8 – Demo Mode & Field-Ready Experience
- Demo mode banner reminds presenters when stimulation is simulated.
- Responsive layout tweaks improve iPad support; journeys section gives pre-built stories for sales teams.

## Phase 9 – QA & Documentation
- Added automated tests (`tests/test_demo_structure.py`) to verify new layout hooks and screens.
- These release notes document the end-to-end scope for stakeholders and demo facilitators.
