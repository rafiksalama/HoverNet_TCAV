# AxioBionics Project Documentation (Archived)

This is an archived copy of the comprehensive project documentation retained from `docs/archive/DOCUMENTATION.md` prior to consolidation into `documentation/`. It is kept for historical reference. Active updates should target inline code comments, `README.md`, or new focused docs under `documentation/`.

(Original content preserved below)

<!-- ARCHIVE BEGIN -->
# AxioBionics Project Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Core Components](#core-components)
6. [Dependencies](#dependencies)
7. [Build System](#build-system)
8. [Database Schema](#database-schema)
9. [Hardware Interface](#hardware-interface)
10. [Bluetooth Communication Protocol](#bluetooth-communication-protocol)
11. [Development Guidelines](#development-guidelines)
12. [Identified Issues](#identified-issues)
13. [Recommendations](#recommendations)

---

## Executive Summary

**AxioNp2 (AxioBionics)** is a medical-grade electrical stimulation therapy platform designed for TENS (Transcutaneous Electrical Nerve Stimulation) and EMS (Electrical Muscle Stimulation) therapy. The application provides comprehensive therapy management with both wireless (Bluetooth) and direct hardware control capabilities.

**Version**: 0.3.15 (Build 30)
**Technology Stack**: Qt6/QML, C++17, SQLite
**Target Platforms**: Linux (embedded), iOS, Android, macOS
**Primary Hardware**: Raspberry Pi Zero 2W with ARM Cortex-M4 coprocessor

**Key Features**:
- 4-node, 8-channel electrical stimulation system
- Dual connectivity (BLE wireless + direct hardware)
- Cloud telemetry via AWS IoT/MQTT
- Comprehensive therapy tracking and analytics
- Multi-platform Material Design UI
- Medical-grade safety features

---

## Project Overview

### Purpose
AxioBionics provides professional-grade electrical stimulation therapy for:
- Chronic pain management
- Muscle rehabilitation
- Neurological therapy
- Physical therapy and recovery
- Spasticity treatment

### Users
- **Patients**: Home-based therapy with intuitive interface
- **Clinicians**: Professional configuration and monitoring tools
- **Administrators**: Cloud-based data analysis and reporting

### Code Statistics
- **C++ Code**: ~6,500 lines
- **QML Files**: 172 files
- **Languages**: C++17, QML, JavaScript
- **CMake Build Files**: Multiple modular configurations

---

## Architecture

### Multi-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Presentation Layer (QML)                    │
│  Material Design UI, Views, Components, Delegates        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         Business Logic Layer (QML + C++)                 │
│  Controllers, Models, State Management, QtQuickStream    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        Device Communication Layer                        │
│  BLE (4 nodes × 2 channels) | Direct Hardware (GPIO/SPI) │
│  Dual-core: A-core (Linux) + M4 (Real-time)             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Cloud/Network Layer                            │
│  MQTT (AWS IoT Core), SSL/TLS, Certificate Auth         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Data Layer                                  │
│  SQLite (Templates, Usages), QtQuickStream Persistence   │
└─────────────────────────────────────────────────────────┘
```

### Runtime Data Flow Overview

1. Startup (`main.cpp`)
   - Qt application & QML engine start; imports registered.
   - `Main.qml` loads; `AppCore` creates default repository and binds `RootModel`.
2. Persistence
   - Configuration JSON deserialized into `AppCore.defaultRepo.qsRootObject`.
   - `UiSession.appModel` binds dynamically to repository root for live updates.
3. Device / Channel Initialization
   - Channels & nodes created if absent; each `ChannelCPP` exposed via `Channel.qml`.
4. User Interaction Loop
   - UI components mutate enabled/intensity; setters validate enum bounds & emit change signals.
5. Hardware / Firmware Bridge
   - Controllers translate logical parameters into stimulation commands (M4Core / GPIO / SPI).
   - Telemetry (volt/current/resistance) updates QML-bound properties.
6. Session Tracking
   - Usage sessions persisted (UsagesSqlAdapter + QtQuickStream) for reporting.
7. Cloud Telemetry
   - MQTT publishes device health & usage summaries; remote adjustments optionally consumed.

### Core Object Relationships
```
UiSession --> RootModel (repository root)
RootModel.device --> Device (I_Device) --> channels[] (ChannelCPP) / nodes[]
RootModel.parameters --> ParametersCPP
RootModel.templatesController --> Template collection
GarmentCPP --> Electrode mapping consumed by Garment views
```

<!-- ARCHIVE END -->
