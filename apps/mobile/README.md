# Brazier Mobile

An Expo SDK 54 companion for iOS and Android. Pair it with a Brazier host that
supports invite preview, then prepare a text model, open conversation history,
and stream text chat. Conversations and completed replies are saved on the
host. Stopping generation keeps the user message but discards the unfinished
reply. This slice does not offer agent, voice, tools, or model management.

## Set up the workspace

From the repository root, use Node 24 and the pnpm version pinned in
`package.json`:

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @brazier/mobile check
pnpm --filter @brazier/mobile test
```

Expo generates the native projects locally. They are not committed.

### Android

Install JDK 17 or 21, Android SDK Platform 36, and Android Build Tools 36.0.0.
Set `ANDROID_HOME` and start an API 35+ emulator or connect a USB-debuggable
device. From the repo root:

```sh
pnpm --filter @brazier/mobile android
```

To build a debug APK separately and run it with Metro:

```sh
pnpm --filter @brazier/mobile exec expo prebuild --platform android --no-install
(cd apps/mobile/android && ./gradlew :app:assembleDebug)
"$ANDROID_HOME/platform-tools/adb" install -r apps/mobile/android/app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/platform-tools/adb" reverse tcp:8081 tcp:8081
pnpm --filter @brazier/mobile start --localhost
```

### iOS

On macOS with Xcode, an iOS Simulator, and CocoaPods installed:

```sh
pnpm --filter @brazier/mobile ios
```

For a physical iPhone, run `pnpm --filter @brazier/mobile ios --device` and
configure a signing team in Xcode if prompted. Paste an invite on a simulator;
use a physical device to test camera scanning.

## Pair and chat

1. Set up a reachable, platform-trusted HTTPS origin for the host. See
   [remote access](../../docs/remote-access.md).
2. In desktop **Manage → Connections**, create a one-time invite with
   **inference** access. Scan the QR or paste **Copy invite** into the mobile app.
3. Review the host, origin, and scopes, then confirm. The app stores its
   credential in Keychain (iOS) or Keystore-backed encrypted storage (Android).
   If the host revokes it, choose **Pair again** and use a new invite.
4. Choose a text model to prepare it for chat. Open a saved conversation or start
   a new one, send a message, and use **Stop** to abort generation.

For a local Android emulator smoke test, a **debug** build also accepts
`http://10.0.2.2:7614` (the emulator's host-loopback alias). Start a local
daemon on port 7614 with authentication enabled, then issue an inference invite
via the authenticated `POST /api/v1/auth/pairings` endpoint. Paste the legacy
format below with its actual ID and one-time code. Do not use a real credential
or a public host with this debug-only HTTP exception.

```text
Daemon URL: http://10.0.2.2:7614
Pairing id: <id>
Pairing code: <code>
```
