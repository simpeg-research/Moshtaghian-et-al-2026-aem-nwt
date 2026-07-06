@echo off
cd /d "%~dp0"
echo === Step 1: Wipe all broken git state ===
rmdir /s /q .git\rebase-merge 2>nul
rmdir /s /q .git\rebase-apply 2>nul
del /f .git\MERGE_HEAD 2>nul
del /f .git\CHERRY_PICK_HEAD 2>nul
del /f .git\index.lock 2>nul

echo === Step 2: Reset index to a clean state ===
git reset --hard origin/main
if %ERRORLEVEL% NEQ 0 (
    echo Hard reset failed. Trying soft reset...
    del /f .git\index 2>nul
    git reset --hard origin/main
)

echo === Step 3: Restore our code changes from original commit ===
git checkout 59e4df0 -- src/plotting.py src/pbtz.py src/gradient.py src/depth_of_investigation.py main.py run_aem_inversion.py environment.yml data/AEM_NWT_PaperLines.txt .gitignore LICENSE "outputs/L150020_fixedbeta_RMS.png"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not restore from commit 59e4df0. Files may already be correct.
)

echo === Step 4: Put resolved README in place ===
copy /y README_resolved.md README.md

echo === Step 5: Commit ===
git config user.name "Keytash Moshtaghian"
git config user.email "keytash.msh97@gmail.com"
git add -A
git status
git commit -m "Fix x-axis to relative km; update citation to in revision #2026AV002398; v1.0.2"

echo === Step 6: Tag and push ===
git tag -d v1.0.2 2>nul
git tag v1.0.2
git push origin main
git push origin v1.0.2 --force
echo.
echo Done! Check output above for errors.
pause
