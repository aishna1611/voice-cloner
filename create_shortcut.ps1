# Creates a Desktop shortcut to run Voice Cloning Studio
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop    = [Environment]::GetFolderPath("Desktop")
$shortcut   = "$desktop\Voice Cloning Studio.lnk"

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($shortcut)
$sc.TargetPath       = "$scriptDir\run_studio.bat"
$sc.WorkingDirectory = $scriptDir
$sc.WindowStyle      = 1
$sc.Description      = "Voice Cloning Studio — Offline TTS"
$sc.Save()

Write-Host "Shortcut created on Desktop: $shortcut" -ForegroundColor Green
