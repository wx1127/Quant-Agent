param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('@sha256:[a-f0-9]{64}$')]
    [string]$CandidateImage,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('@sha256:[a-f0-9]{64}$')]
    [string]$PreviousImage
)

$ErrorActionPreference = 'Stop'
$composeFile = Join-Path $PSScriptRoot '../../deploy/compose.test.yml'

& (Join-Path $PSScriptRoot 'deploy.ps1') -ImageRef $CandidateImage -Environment test
if ($LASTEXITCODE -ne 0) {
    throw 'Candidate test deployment failed.'
}

$auditMarker = docker compose --file $composeFile exec -T api python -c `
    "from pathlib import Path; import json; p=Path('/var/lib/quant-agent/audit/events.jsonl'); p.parent.mkdir(parents=True, exist_ok=True); print(p.open('a', encoding='utf-8').write(json.dumps({'event':'deployment_acceptance_marker'}, separators=(',', ':')) + '\n'))"
if ($LASTEXITCODE -ne 0) {
    throw 'Could not append the deployment acceptance audit marker.'
}

$before = docker compose --file $composeFile exec -T api python -c `
    "from pathlib import Path; p=Path('/var/lib/quant-agent/audit/events.jsonl'); print(len(p.read_text(encoding='utf-8').splitlines()) if p.exists() else 0)"
if ($LASTEXITCODE -ne 0) {
    throw 'Could not inspect the append-only audit volume.'
}
if ([int]$before -lt 1) {
    throw 'Acceptance requires at least one persisted audit record before rollback.'
}

# Simulate a failed candidate process, then prove the prior digest can restore service.
docker compose --file $composeFile stop api
& (Join-Path $PSScriptRoot 'rollback.ps1') -ImageRef $PreviousImage -Environment test
if ($LASTEXITCODE -ne 0) {
    throw 'Rollback command failed.'
}

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health/ready' -TimeoutSec 3
        if ($response.status -eq 'ready') {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $ready) {
    throw 'Rollback did not restore a healthy service.'
}

$after = docker compose --file $composeFile exec -T api python -c `
    "from pathlib import Path; p=Path('/var/lib/quant-agent/audit/events.jsonl'); print(len(p.read_text(encoding='utf-8').splitlines()) if p.exists() else 0)"
if ([int]$after -lt [int]$before) {
    throw 'Audit record count decreased across rollback.'
}
Write-Output "Deployment and rollback acceptance passed; audit records before=${before}, after=${after}."
