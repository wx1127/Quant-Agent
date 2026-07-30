param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('@sha256:[a-f0-9]{64}$')]
    [string]$ImageRef,
    [ValidateSet('test', 'production')]
    [string]$Environment = 'test'
)

$ErrorActionPreference = 'Stop'
$composeFile = Join-Path $PSScriptRoot "../../deploy/compose.${Environment}.yml"
$env:QUANT_AGENT_IMAGE = $ImageRef

# Rollback is code-only: never run Alembic downgrade and never replace audit volumes.
docker compose --file $composeFile config --quiet
docker compose --file $composeFile up --detach --no-deps api
if ($LASTEXITCODE -ne 0) {
    throw 'Rollback image failed to start.'
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
if (-not $healthy) {
    throw 'Rollback image did not become healthy.'
}
Write-Output "Rollback healthy with immutable image ${ImageRef}; schema and audit volume retained."
