# Documentation Index

Central index for all project documentation. Historical / verbose documents live under `documentation/archive/`. This file maps categories to archived documents with brief purpose notes.

## Active Entry Points
- README.md: High-level overview, build/run basics.
- documentation/INDEX.md: (this file) Navigational map.

## Architecture & Core
- archive/DOCUMENTATION.md: System architecture, data flow, lifecycle, components.
- archive/ENVIRONMENT_VERIFIED.md: Notes confirming environment/toolchain verification.

## Build & Setup
- archive/ANDROID_SETUP.md: Detailed Android environment preparation.
- archive/ANDROID_QUICKSTART.md: Condensed Android build/run steps.
- archive/APPLE_DEVELOPER_SETUP.md: Apple developer account & certificates guidance.
- archive/IOS_SETUP.md: iOS toolchain and provisioning steps.
- archive/SETUP_SUMMARY.md: Cross-platform summary of essential setup steps.
- archive/SETUP_STATUS.md: Checklist/status markers for setup stages.
- archive/SETUP_COMPLETE.md: Marker & criteria for completed setup.

## Regulatory & Risk (iOS focus)
- archive/IOS_REGULATORY_CHECKLIST.md: Regulatory compliance checklist items.
- archive/IOS_RISK_ANALYSIS.md: Risk analysis summary and mitigations.

## Phases & Progress
- archive/PHASE_0_SUMMARY.md: Phase 0 outcomes & lessons.
- archive/PHASE_1_QUICKSTART.md: Phase 1 quickstart goals & directions.
- archive/CURRENT_PROGRESS.md: Snapshot of ongoing progress (historical).
- archive/BUILD_STATUS.md: Historical build status reporting.
- archive/INSTALLATION_STATUS.md: Installation verification notes.

## Platform Notes
- archive/QT_ANDROID_NOTES.md: Qt + Android integration caveats.

## Issues & Tracking
- archive/ISSUES.md: Historical issue log / references.

## Navigation Guidelines
1. Prefer README.md and concise in-source documentation for day-to-day development.
2. Use archived documents only for deep dives or historical context; do not update them for routine changes.
3. Add new long-form documents directly to `documentation/` (not root) and consider whether they belong in `archive/` from inception.
4. Keep this index updated when adding or retiring documents.

## Adding a New Document
1. Place concise, actively maintained docs in `documentation/`.
2. Place large, historical, or regulatory docs in `documentation/archive/`.
3. Add an entry here with: filename, short purpose (one line), category.
4. Update README.md if it is a primary entry point.

## Planned Additions
- Logging categories reference (future).
- Security key management & rotation guide.
- Database migration strategy document.

