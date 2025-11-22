# AxioBionics (AxioNp2)

AxioBionics (AxioNp2) is a multi-platform (Linux Embedded, iOS, Android, macOS) electrical stimulation therapy application supporting TENS/EMS modalities with real-time telemetry, template management, garment configuration, and cloud connectivity.

## Quick Start

```bash
git clone <repo-url>
cd AxioBionics
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/AxioNp2
```

## Features
- 8 stimulation channels / 4 nodes
- TENS & EMS modes (see `AppSpec.*`)
- Persistent template & usage logging (QtQuickStream + SQLite)
- BLE device discovery & simulation layer
- MQTT telemetry (AWS IoT)
- Material Design QML UI

## Project Structure (Summary)
- `Src/` C++ backend (channels, parameters, garment, hardware abstraction)
- `Qml/` UI + business logic controllers
- `Ext/` Submodules: QtQuickStream (serialization), QuickBluetooth (BLE)
- `Res/` Fonts, images, SQL schema, sounds
- `Keys/` Device certificates (remove from VCS; see security note)

## Build Notes
- Requires Qt 6.5+ (Quick, Widgets, QuickControls2, Sql, Charts, Bluetooth, Mqtt (non-iOS))
- iOS uses Paho MQTT C++ instead of Qt MQTT
- OpenSSL path for iOS resolved via `OPENSSL_ROOT_DIR` or vcpkg; configure env var if needed.

## Security Notice
Private keys & certificates must NOT reside in version control. Credentials have been removed from `Keys/`. Use environment variables (or platform keychain/keystore) to supply paths at runtime. See `documentation/security-key-management.md` for rotation and storage strategy.

## Documentation
## Documentation

Use `documentation/INDEX.md` for a complete categorized map of all documentation. Historical and verbose material resides in `documentation/archive/`. Active day-to-day references remain inline (C++/QML) and in this README.

Key starting points:
- Architecture overview: `documentation/archive/DOCUMENTATION.md`
- Platform setups: See Index (Android/iOS entries)
- Regulatory & risk (historical): See Index

When adding new long-form docs, place them in `documentation/` (or `documentation/archive/` if primarily historical) and update `documentation/INDEX.md`.

Use this README for high-level orientation; consult archive for deep details. Future updates should prefer concise living docs here and versioned historical material under `docs/archive/`.

## License
Add a license file (e.g., Apache-2.0, MIT, or proprietary) before distribution.

## Contributing
1. Create feature branch
2. Add tests where feasible (Qt Test planned)
3. Run clang-tidy / cppcheck locally
4. Open PR with summary and risk assessment

---
Generated initial README enhancement as part of cleanup consolidation.
