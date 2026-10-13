# options.ps1 - the options menu for this bot (options.py edits config.json).
# Uses the bot's own venv when there is one, and installs pygame into it if missing.
Set-Location $PSScriptRoot

$py = $null
foreach ($v in "venv", "venv311", ".venv", "scripts\.venv") {
    if (-not $py -and (Test-Path "$v\Scripts\python.exe")) { $py = "$v\Scripts\python.exe" }
}
$pyArgs = @()
if (-not $py) { $py = "py"; $pyArgs = @("-3") }

& $py @pyArgs -c "import pygame" 2>$null
if ($LASTEXITCODE -ne 0) { & $py @pyArgs -m pip install pygame }
& $py @pyArgs options.py @args
if ($LASTEXITCODE -ne 0) { Read-Host "Press Enter to close" }
