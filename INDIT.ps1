# ====================================================================
# Kreatív Diktáló - Indító (admin, .venv)
# Dupla kattintás helyett: jobb klikk > "Run with PowerShell"
# vagy terminálból: powershell -ExecutionPolicy Bypass -File INDIT.ps1
# ====================================================================

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Admin jogok ellenőrzése / kérése (globális F8 hotkeyhez kell)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Admin jogok kerese (UAC ablak: kattints IGEN)..." -ForegroundColor Yellow
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Set-Location $projectDir
$env:PYTHONIOENCODING = "utf-8"

$py = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[HIBA] Nincs .venv! Elobb: py -3.14 -m venv .venv" -ForegroundColor Red
    Read-Host "ENTER a kilepeshez"; exit 1
}

Write-Host "Kreativ Diktalo indul (admin)..." -ForegroundColor Green
Write-Host "F8 = diktalas barhol | Ctrl+Shift+Space = parancs mod" -ForegroundColor Cyan
& $py "src\gui_main.py"

Read-Host "ENTER a kilepeshez"
