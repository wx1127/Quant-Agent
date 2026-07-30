param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('@sha256:[a-f0-9]{64}$')]
    [string]$ImageRef,
    [ValidateSet('test', 'production')]
    [string]$Environment = 'test',
    [string]$PreviousImageRef = ''
)

$ErrorActionPreference = 'Stop'
$composeFile = Join-Path $PSScriptRoot "../../deploy/compose.${Environment}.yml"
$env:QUANT_AGENT_IMAGE = $ImageRef

docker compose --file $composeFile config --quiet
if ($LASTEXITCODE -ne 0) {
    throw 'Compose configuration validation failed.'
}
docker compose --file $composeFile run --rm migrate
if ($LASTEXITCODE -ne 0) {
    throw 'Forward migration failed; candidate was not started.'
}
docker compose --file $composeFile up --detach api
if ($LASTEXITCODE -ne 0) {
    if ($PreviousImageRef -match '@sha256:[a-f0-9]{64}$') {
        & (Join-Path $PSScriptRoot 'rollback.ps1') `
            -ImageRef $PreviousImageRef -Environment $Environment
    }
    throw 'Candidate startup failed.'
}

$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health/ready' -TimeoutSec 3
        if ($response.status -eq 'ready') {
            $healthy = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}
if ($healthy) {
    Write-Output "Deployment healthy: ${ImageRef}"
    exit 0
}
if ($PreviousImageRef -match '@sha256:[a-f0-9]{64}$') {
    & (Join-Path $PSScriptRoot 'rollback.ps1') -ImageRef $PreviousImageRef -Environment $Environment
}
throw 'Candidate health check failed; rollback was attempted when a previous digest was supplied.'
