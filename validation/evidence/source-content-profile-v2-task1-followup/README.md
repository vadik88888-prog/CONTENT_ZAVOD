# Source Content Profile v2 Task 1 follow-up — real-window QA

- Result: `PASS`
- Captured: `2026-08-14T15:04:16.088684+00:00`
- Host: Windows 10.0.26200, Qt platform plugin `windows`
- Window: real top-level `ProjectScreen`, native window id recorded in `runtime-evidence.json`
- Display: `\\.\DISPLAY1`, 1536×816 logical pixels, DPR 1.25
- Screenshot: 1750×920 physical pixels

The advanced-settings panel was opened through its real UI toggle. The window
was scrolled until all four Source Content Profile controls were fully inside
the visible scroll viewport. The harness then selected and persisted:

- format: `gameplay` (`Геймплей`)
- editorial mode: `commentary` (`Комментарий`)
- domain: `gaming` (`Игры`)
- traits: `visual_led` (`Ведёт визуал`)

For every control, runtime inspection confirmed that the complete ordered IDs
and labels match `content_profile_taxonomy`, the control is visible and enabled,
and the interactive selection was persisted by the real `ProjectViewModel` and
`DesktopProjectStore` flow.

Evidence:

- `project-screen-profile-controls.png` — SHA-256
  `157d747c7adf69bbd8156aa28a3ca8c0334872e514addb9ed0987242cac49b44`
- `runtime-evidence.json` — SHA-256
  `565ddfd8bf0c42c56a3f88da17760e961bbe2beeea26fb0d57d419cc696d2661`

Reproduction command from the repository root:

```powershell
$env:QT_QPA_PLATFORM='windows'
.\.venv\Scripts\python.exe validation\source_content_profile_real_window_qa.py validation\evidence\source-content-profile-v2-task1-followup
```
