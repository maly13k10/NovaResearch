# Windows Executable & Desktop Icon

## One-click local build
Run `build_windows.bat` on a Windows PC with internet access.

Output:
- `dist\NovaResearchAI.exe`
- Desktop shortcut `NovaResearch AI.lnk`
- `professor_package\NovaResearchAI_Professor_Windows.zip`

The application executable launches a local Streamlit server on a free port and opens the browser automatically. The Nova `.ico` file is embedded into the executable.

## Cloud build without configuring PyInstaller locally
1. Create a GitHub repository.
2. Upload this project.
3. Open **Actions** → **Build Windows EXE** → **Run workflow**.
4. Download the `NovaResearchAI-Windows` artifact.

A Windows executable must be built on a Windows environment. The included GitHub Actions workflow provides that environment automatically.
