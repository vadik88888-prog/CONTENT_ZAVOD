# Friend Beta activation signing

The desktop app displays a device code on its activation screen. On the
administrator machine, create a signed licence outside the repository:

```powershell
.\.venv\Scripts\python.exe tools\sign_friend_beta_license.py `
  --device-code "CFB1-..." `
  --expires 2026-12-31 `
  --output "$env:USERPROFILE\Desktop\friend-beta-license.json"
```

Give that `.json` file to the Friend Beta user. They select it on the
activation screen. The code is device-bound; the same file is rejected on a
different PC.

The signing seed lives only at
`%LOCALAPPDATA%\ContentFactoryAdmin\friend_beta_signing.seed`. Back it up
securely and never copy it into this repository, a build folder, or a ZIP.
