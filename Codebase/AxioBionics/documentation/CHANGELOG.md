# Changelog

## Unreleased (Documentation & Code Quality Consolidation)

### Overview
This merge consolidates documentation, improves code safety, enhances security practices, and cleans repository structure. Root markdown sprawl has been archived into `documentation/archive/`, leaving `README.md` as the single root doc entry point. SSL credential handling has been refactored to use environment variables with fingerprint and expiration logging.

### Documentation Restructure
- Centralized all active docs under `documentation/` with historical files in `documentation/archive/`.
- Added `INDEX.md` for navigable entry points.
- Added `security-key-management.md` covering certificate rotation, storage, and auditing.
- Created this `CHANGELOG.md` to track consolidated improvements.

### Code Quality Improvements
- Fixed enum and property naming inconsistencies in core C++ classes (`AppSpecCpp`, `GarmentCPP`, `ParametersCPP`).
- Corrected missing/incorrect NOTIFY signals and initialization defaults.
- Removed unsafe implicit QObject copy semantics.
- Added/expanded Doxygen-style blocks and QML inline documentation for better API clarity.

### Security & TLS Enhancements
- Removed committed certificate/key artifacts; added ignore patterns to prevent future accidental commits.
- Introduced environment-variable-based SSL credential loading.
- Added SHA256 fingerprint and expiration date logging for loaded certificates.
- Documented rotation procedure and auditing strategy.

### Repository Hygiene
- Archived legacy markdown files (Android/iOS setup, progress logs, phase summaries) to reduce top-level clutter.
- Cleaned `.gitignore` to only allow `README.md` at root and exclude sensitive material.

### Pending / Future Work
- Optional abstraction: introduce a `SecretProvider` interface for platform-specific credential resolution.
- Automate periodic certificate re-validation (scheduled task or startup hook).
- Evaluate converting deep historical archive docs to PDF for immutable retention.

### Merge Intent
This changelog accompanies the branch merge to provide a clear, human-readable audit trail of structural, documentation, and security changes.
