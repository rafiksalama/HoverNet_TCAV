# Android Development Setup for AxioBionics (Archived)

Migrated from root. Use README + security docs for active guidance; retain full historical setup here.

<!-- ORIGINAL CONTENT BELOW -->
# Android Development Setup for AxioBionics

This guide will help you set up Android development environment, build the APK, and run it on an emulator.

## Current Environment Status

✅ **Installed**:
- Homebrew: 4.6.20
- CMake: Available at `/opt/homebrew/bin/cmake`
- macOS: Darwin 24.6.0 (Apple Silicon - arm64)

❌ **Not Installed**:
- Android Studio / Android SDK
- Android Command Line Tools
- Java JDK
- Qt 6 for Android
- Android Emulator

---

## Prerequisites Installation

### Step 1: Install Java JDK (Required for Android)

```bash
brew install openjdk@17
echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
java -version
```

### Step 2: Install Android Command Line Tools
...
