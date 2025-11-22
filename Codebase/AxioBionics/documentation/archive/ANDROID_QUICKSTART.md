# Android Quick Start Guide (Archived)

Migrated from project root. Active quick start should be summarized in README; keep this for historical detail.

<!-- ORIGINAL CONTENT BELOW -->
# Android Quick Start Guide

Get AxioBionics running on Android emulator in 3 easy steps!

## Prerequisites

- macOS (Apple Silicon or Intel)
- Homebrew installed
- ~10 GB free disk space
- 30-60 minutes for initial setup

## Step 1: Setup Android Environment

Run the automated setup script:

```bash
cd /Users/rafik.salama/Codebase/AxioBionics
./android-setup.sh
```

This will install:
- Java JDK 17
- Android SDK & NDK
- Android Emulator
- Create virtual device (Pixel 6)

**Then restart your terminal** or run:
```bash
source ~/.zshrc
```

## Step 2: Install Qt 6 for Android

Download Qt Online Installer:
https://www.qt.io/download-qt-installer

During installation, select:
- ☑ Qt 6.5.x (or later)
- ☑ Android ARM64-v8a
- ☑ Qt Quick Controls 2
- ☑ Qt Charts
- ☑ Qt MQTT
- ☑ Qt Bluetooth

Installation location: `~/Qt` (default)

## Step 3: Build and Run

```bash
# Build APK
./android-build.sh

# Deploy to emulator and test
./android-test.sh
```

That's it! The app should launch on the emulator.

## Troubleshooting

### "Qt not found"
Install Qt from https://www.qt.io/download-qt-installer

Or specify manually:
```bash
export QT_ANDROID_PATH=~/Qt/6.5.3/android_arm64_v8a
export QT_HOST_PATH=~/Qt/6.5.3/macos
./android-build.sh
```

### "Build failed"
Check logs:
```bash
cat build-android/cmake-configure.log
cat build-android/cmake-build.log
```

### "Emulator won't start"
List available AVDs:
```bash
avdmanager list avd
```

Start manually:
```bash
emulator -avd AxioBionics_Pixel6 -gpu host
```

### "App crashes on launch"
View logs:
```bash
adb logcat | grep -i "axio\|qt"
```

## Manual Commands

### Build only
```bash
./android-build.sh
```

### Clean build
```bash
./android-build.sh --clean
```

### Install manually
```bash
adb install -r AxioNp2.apk
```

### Launch app
```bash
adb shell am start -n org.axiobionics.axionp2/.MainActivity
```

### Stop emulator
```bash
adb emu kill
```

## What to Test

Once running, verify:
- [ ] App launches without crash
- [ ] Splash screen displays
- [ ] Login screen appears
- [ ] Navigation works
- [ ] Can create/edit templates
- [ ] Settings menu accessible
- [ ] BLE simulation mode works
- [ ] UI renders correctly

## File Locations

- APK: `AxioNp2.apk`
- Build directory: `build-android/`
- Logs: `build-android/*.log`
- Emulator: `~/.android/avd/AxioBionics_Pixel6.avd/`

## Resources

- Full guide: [ANDROID_SETUP.md](ANDROID_SETUP.md)
- Project docs: [DOCUMENTATION.md](DOCUMENTATION.md)
- Issues list: [ISSUES.md](ISSUES.md)
