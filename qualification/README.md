# Beta qualification evidence

`beta-manifest.json` is the release contract for the hardware-dependent paths
that hosted unit tests cannot prove. A beta tag is qualified only when its
commit has one passing result for every package and voice host named there.

Result files live outside the source tree while a run is in progress and are
published as workflow artifacts. Before a tag is published, the release job
downloads them into one directory and runs:

```sh
node scripts/verify-beta-qualification.mjs --results qualification-results --commit "$GITHUB_SHA"
```

Every result is JSON with `schema_version: 1`, `commit`, `kind`, and a stable
host/artifact identity. Voice results also contain corpus provenance, microphone
class, and the measured fields named in `voice_budgets`. Package results record
that the installed artifact started its bundled daemon, contained the Computer
safety helper, loaded the Pi worker and its dependency closure, opened and
deleted a no-model agent session, stopped the worker, and waited for the daemon
to exit. The Windows report additionally proves that the installed
AppContainer launcher passed its real junction/tool-access isolation probe.

The verifier deliberately refuses missing, stale, duplicated, or failing
evidence. Brazier has no telemetry; operators explicitly upload these local
qualification artifacts to the release workflow.

## Voice hardware protocol

Use the **Qualify voice** control in a live Voice session. The app snapshots
the live metrics; the operator does not type timing or detector results.
Run the exact candidate build with `BRAZIER_BUILD_COMMIT` set to its full
`git rev-parse HEAD` value; packaged release-candidate builds embed the same
value automatically. Version-only or dirty/unidentified builds cannot produce
evidence accepted for a tag.

1. Select the actual microphone class and start the trial. Read each of the 20
   displayed sentences once, in order. During at least three assistant answers,
   interrupt naturally so stop latency has real samples.
2. Start the noise window. Do not speak for at least five minutes while exposing
   the microphone to ordinary room sound, fan noise, and normal keyboard use.
   Any captured utterance is counted as a false sustained interruption. The
   deterministic synthetic corpus remains a separate, repeatable model gate.
3. Save the result. The app hashes the fixed speech corpus, records host memory,
   GPU memory where available, model ids, raw sample counts, percentile metrics,
   and the exact build commit. A fallback energy detector or incomplete sample
   set produces `passed: false` and cannot qualify a release.

Run the protocol once on the Apple-Silicon host and once on the Linux/NVIDIA
host named in the manifest. Headphones avoid making the assistant's own voice
an accidental noise-test input.

`voice/synthetic-noise-corpus.json` is the checked-in deterministic regression
set. `pnpm qualification:voice-corpus` runs the shipped Silero ONNX model (not a
fake) against silence, room noise, hum, keyboard-like impulses, and background
harmonics while measuring inference and simulated queue lag. Passing it is
necessary but not hardware qualification: microphone capture, speech recall,
echo, ASR latency, and interruption latency still require the explicit
`voice-hardware` artifacts above.
