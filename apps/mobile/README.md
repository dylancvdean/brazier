# Brazier Mobile

An Expo SDK 54 companion for iOS and Android. Pair it with one or more Brazier
hosts that support invite preview, then prepare a text model, open conversation
history, and stream text chat. The tools menu in the composer chooses which of
the host's tools (web search, page fetch, MCP servers and the rest) the model
may call; tool calls and their results show in the conversation. Conversations and completed replies are saved on
the host. Stopping generation keeps the user message but discards the
unfinished reply. Settings holds the theme (system, dark, light) and the paired
hosts; each host keeps its own credential and pinned certificate.

## Set up the workspace

From the repository root, use Node 24 and the pnpm version pinned in
`package.json`:

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @brazier/mobile check
pnpm --filter @brazier/mobile test
pnpm --filter @brazier/mobile llama:artifacts
```

`llama:artifacts` downloads the prebuilt llama.cpp libraries for `llama.rn`
(an iOS xcframework and Android `.so` files) and checks them against the
SHA-256 values in the package. The package would do this in its install script,
which this workspace does not run. Repeat it after `llama.rn` is reinstalled;
native builds fail without it.

The iOS project is committed for Xcode Cloud. Expo generates the Android
project locally; it is not committed.

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

### TestFlight with Xcode Cloud

The iOS app uses Apple's Xcode Cloud, under personal team `4U2WZ7WJT4`.
Its App Store Connect listing is [Brazier Mobile](https://appstoreconnect.apple.com/apps/6839438555/testflight)
(`ai.brazier.mobile`); the installed app displays **Brazier**.
No Expo account or EAS token is required.

Open `apps/mobile/ios/Brazier.xcworkspace` after installing the workspace
packages and running `pod install --deployment` in `apps/mobile/ios`.
The shared scheme is `Brazier`. In Xcode, choose **Integrate → Create Workflow**,
select **Brazier**, and connect this GitHub repository.
Configure the workflow as follows:

- Name: **Master → TestFlight**.
- Start condition: **Branch Changes**, with the exact branch **master**.
- Files and Folders: **Custom Rules → Start a build**, with **Any File**
  under `/apps/mobile` or `/packages/pairing` (recursive). These are the mobile
  app and its shared invite code; unrelated desktop, Rust, and documentation
  changes do not start TestFlight builds. Use a manual start for root dependency
  or workspace configuration changes that affect mobile without changing either
  of these folders.
- Action: **Archive**, iOS, scheme **Brazier**, configuration **Release**.
- Distribution preparation: **TestFlight (Internal Testing Only)**.
- Post-action: **TestFlight Internal Testing**, group **Brazier Internal**.
- Use the latest stable Xcode/macOS image compatible with the app.

The connected [Master → TestFlight workflow](https://appstoreconnect.apple.com/teams/e3cc8a33-1ac0-4665-98c9-31cd24d0f31c/xcode-cloud/products/5a1c8513-5003-4686-a295-5ac62e266e80/workflows/92686A25-B109-4793-B628-F736D591076B)
uses `master` for both code-push and manual starts. Apple stores these settings
in Xcode Cloud, not a GitHub Actions YAML file.
The committed `ios/ci_scripts/ci_post_clone.sh` installs Node 24 and pinned
pnpm, checks the mobile app, and installs locked CocoaPods dependencies.
`ci_pre_xcodebuild.sh` uses Xcode Cloud's build number so each TestFlight
upload is unique. Apple manages cloud signing; desktop notarization secrets
are unrelated to iOS signing.

The native iOS project, shared scheme, and Podfile.lock are committed so Xcode
Cloud can discover the app before running scripts. After changing Expo native
configuration or native dependencies, regenerate and review the project:

```sh
pnpm --filter @brazier/mobile exec expo prebuild --platform ios --no-install
(cd apps/mobile/ios && pod install)
```

Keep `ios/ci_scripts` and the shared scheme when regenerating; avoid
`expo prebuild --clean`, which deletes that directory. Commit native changes
and the updated lockfile. Do not commit `Pods`, `.xcode.env.local`, signing
credentials, or Xcode user data. TestFlight installs include the JavaScript
bundle and do not need Metro. External testers may require Beta App Review;
App Store publication requires a separate App Review submission.

The local `plugins/with-ios-scenes.cjs` config plugin moves React Native startup
to `SceneDelegate` and preserves that change during prebuild. Its Swift source
is `plugins/SceneDelegate.swift`; the appended copy in `AppDelegate.swift` is
generated. `app.json` supplies the scene manifest. This is required when building
with the iOS 27 SDK: UIKit terminates apps using only the old app delegate
lifecycle before they display a screen. Scene callbacks also forward lifecycle
events and links to the Expo app delegate.

The **Mobile iOS** GitHub workflow builds and launches a Release simulator app
for mobile changes, checking that it stays alive without Metro. To repeat the
launch check locally after building a Release simulator app:

```sh
bash apps/mobile/scripts/smoke-ios.sh <simulator-udid> <path-to-Brazier.app>
```

## Models on this device

**This device** sits next to the paired hosts in Settings and works with no
host paired. Its chats are text only: tools, web access, Generate, Agent and
voice need a host and are not shown. Conversations are saved in the app's
documents on the phone, and so are the models.

Open **Models** to add one: type a Hugging Face repository as `owner/name`,
choose **List files**, and download a GGUF file. The screen shows each file's
size and the free space, and lists what is stored with a **Delete** action.
Only public repositories work; there is no search.

GGUF models run through llama.cpp (`llama.rn`, pinned to the latest stable
release, 0.12.9), using Metal on an iPhone and the CPU in the iOS Simulator.
`src/local/engine.ts` is the interface Chat talks to; a paired host and this
device both implement it, and each on-device runtime is a `Runtime` behind it.

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
