# Génère, sépare, fenêtre et valide les jeux synthétiques v1 (hors Git, sous data\).
# Usage : .\scripts\prepare_synthetic_data.ps1 [-Overwrite]
# Ne lance aucun entraînement et ne touche pas aux poids.
param([switch]$Overwrite)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$ow = @()
if ($Overwrite) { $ow = @("--overwrite") }

$sets = @(
    @{ id = "synthetic_vibration_v1";       window = 2048; hop = 1024 },
    @{ id = "synthetic_acoustic_events_v1"; window = 0;    hop = 0 },
    @{ id = "synthetic_slow_series_v1";     window = 0;    hop = 0 }
)

foreach ($s in $sets) {
    $root = "data\$($s.id)"
    Write-Host "== $($s.id)"
    & $py -m signal_lm.data.cli generate --config "configs\data\$($s.id).json" --root $root @ow
    & $py -m signal_lm.data.cli split --root $root --config configs\data\split_v1.json
    if ($s.window -gt 0) {
        & $py -m signal_lm.data.cli window --root $root --window-samples $s.window --hop-samples $s.hop
    }
    & $py -m signal_lm.data.cli validate --root $root --regenerate --strict
}

Write-Host "Rapports : data\<jeu>\quality_report.md"
