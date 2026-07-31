# CONTENT FACTORY — OUTPUT QUALITY STANDARD

**Status:** Target Quality Source of Truth — not current pipeline behavior

## 0. Current implementation versus target behavior

`PASS`, `PASS_WITH_WARNINGS` and `BLOCKED` are target product-quality statuses.
They are not implemented readiness statuses in the current pipeline.

Today, the operational lifecycle uses `completed`, `warning`,
`completed_with_warnings` and `failed`. These states describe pipeline/render
execution and recovery; they must not be presented as, or translated into, the
target quality statuses.

`QualityReport` becomes the owner of final readiness only after Goal 5G. Until
that Goal is implemented, existing technical validation and pre-render quality
warnings remain useful checks, but they do not implement this QualityReport or
the `PASS` / `PASS_WITH_WARNINGS` / `BLOCKED` aggregation.

## 1. Target quality result (after Goal 5G)

```text
PASS
PASS_WITH_WARNINGS
BLOCKED
```

Aggregation:

```text
if any hard_blocker:
    status = BLOCKED
elif any warning:
    status = PASS_WITH_WARNINGS
else:
    status = PASS
```

A weighted average never overrides a blocker.

## 2. Hard blockers

```text
MEDIA_INVALID
DURATION_MISMATCH
BLACK_FRAME_CRITICAL
AUDIO_SILENT_CRITICAL
AUDIO_CLIPPING_CRITICAL
BOUNDARY_WORD_CUT
SEMANTIC_INCOMPLETE
CONTEXT_DEBT_CRITICAL
FACE_CROP_CRITICAL
TARGET_MISSING
SUBTITLE_OUT_OF_FRAME
SUBTITLE_UNREADABLE
LICENSE_BLOCKED
WRONG_ARTIFACT_LINK
DUPLICATE_OUTPUT
SOURCE_IDENTITY_MISMATCH
EDIT_PLAN_MISMATCH
AUDIO_VIDEO_DESYNC_CRITICAL
PLATFORM_MASK_CRITICAL
```

## 3. Warnings

```text
DURATION_VARIANCE_LOW
BLACK_FRAME_BRIEF
AUDIO_NOISE_HIGH
AUDIO_LOUDNESS_OUTSIDE_TARGET
BOUNDARY_ABRUPT
CONTEXT_DEBT_MEDIUM
FACE_CROP_RISK
TARGET_CONFIDENCE_LOW
CROP_MOTION_HIGH
CROP_SWITCH_FREQUENT
EMPTY_FRAME_BRIEF
SUBTITLE_CPS_HIGH
SUBTITLE_LINE_BREAK_WEAK
SUBTITLE_OVERLAP_RISK
FONT_FALLBACK_USED
NEAR_DUPLICATE_OUTPUT
PROVISIONAL_PLATFORM_MASK
LOCAL_FALLBACK_USED
QUALITY_CONFIDENCE_LOW
```

## 4. Technical checks

Validate:

- container;
- expected streams;
- codecs;
- width/height;
- frame rate;
- pixel format;
- duration;
- file size;
- checksum;
- stream errors.

Default export contract:

```text
MP4
H.264
AAC-LC
progressive
square pixels
yuv420p
1080×1920
fixed 30 fps
```

## 5. Boundary quality

Hard:

- no word cut;
- no missing payoff;
- no truncated answer;
- valid source interval.

Perceptual:

- no hidden-context continuation;
- intact first phoneme;
- natural ending;
- no audio click;
- filler removal remains natural.

## 6. Semantic quality

Minimum structure:

```text
subject/context
+ meaningful claim/action
+ answer/result/payoff
```

Block when:

- answer lacks required question/context;
- entities are unresolved;
- hook is not delivered;
- clip ends before result;
- main point duplicates another output;
- metadata/title describes absent content.

## 7. Duplicate quality

Evaluate before selection, after boundary refinement and after render.

A duplicate does not become unique because crop or subtitle color differs.

## 8. Composition quality

Metrics:

```text
face_clipping_ratio
headroom_ratio
target_occupancy
speaker_target_match
crop_velocity
crop_acceleration
crop_switch_frequency
salient_object_visibility
empty_frame_ratio
subtitle_overlap_ratio
platform_ui_overlap_ratio
```

Fallback:

```text
stable target crop
→ safe group/wide crop
→ screen/object-first layout
→ designed padded canvas
→ BLOCKED
```

## 9. Subtitle quality

Hard rules:

- max two lines;
- no out-of-frame;
- no critical mask overlap;
- no unreadable CPS;
- no missing required captions;
- no early punchline reveal;
- no severe timing drift.

Initial RU profile:

- target `13–17 CPS`;
- brief maximum `20 CPS`;
- phrase minimum about `0.8–1.0 s`;
- phrase usually `4–6 s` max;
- syntax-aware line breaks;
- validate actual rendered bounding boxes.

## 10. Audio quality

Measure:

- integrated loudness;
- true peak;
- max short-term;
- dialogue loudness;
- clipping;
- silence;
- AV sync;
- intelligibility/noise proxies.

Initial target:

```text
around −16 LUFS
preferred range −18…−14 LUFS
true peak ceiling −1 dBTP
```

Block severe clipping, missing speech, major desync, SFX masking key speech or music masking dialogue.

## 11. Artifact identity

Pass only when:

- path exists;
- file is readable;
- checksum captured;
- all IDs match;
- duration matches;
- expected output belongs to run;
- UI resolves by artifact ID;
- same-named outputs are not overwritten.

`WRONG_ARTIFACT_LINK` is always a blocker.

## 12. Licensing

Every asset stores:

- asset ID;
- fingerprint;
- license;
- source;
- commercial-use permission;
- redistribution permission;
- attribution;
- platform restrictions.

Unknown license → `LICENSE_BLOCKED`.

## 13. Target QualityReport contract (Goal 5G)

Must include:

- schema/report/artifact/project/source/candidate/edit-plan/render IDs;
- status;
- checks;
- metrics;
- fallbacks;
- created timestamp.

Each check includes:

- code;
- severity;
- interval;
- measured value;
- threshold;
- config version;
- auto-fix availability;
- user message;
- technical details.

## 14. Auto-fix policy

Allowed only when deterministic and bounded.

Examples:

- safe pre/post-roll;
- split subtitle;
- reduce font within limits;
- safe crop fallback;
- reduce music gain;
- remove unlicensed optional asset;
- rerender affected candidate only.

Auto-fix creates new edit-plan revision, preserves old report, rerenders downstream only and never silently changes semantics.

## 15. Execution order

```text
identity/provenance
→ media validation
→ duration/streams
→ boundaries
→ semantic completeness
→ duplicates
→ composition
→ subtitles
→ audio
→ platform masks
→ aggregate status
```

## 16. Regression requirements

Minimum cases:

- word cut;
- incomplete answer;
- semantic duplicates;
- face at edge;
- empty-table crop;
- rapid switches;
- subtitle overflow;
- high RU CPS;
- silent speech;
- same-named outputs;
- wrong metadata;
- interrupted render;
- Cyrillic path;
- rerender without reanalysis.

## 17. Target Definition of Done (after Goal 5G)

- status derives from persisted checks;
- every blocker has code/evidence;
- blocker cannot be hidden by score;
- artifact mismatch is detected;
- duplicates block by default;
- crop failures are detected;
- subtitle geometry validates after render;
- rerender affects downstream only;
- UI explains actionable issues;
- real Windows smoke confirms results.
