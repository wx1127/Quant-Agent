param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root '.env'
if (Test-Path -LiteralPath $envFile) {
  Get-Content -LiteralPath $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and !$line.StartsWith('#') -and $line.Contains('=')) {
      $parts = $line.Split('=', 2)
      [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
  }
}
if (![bool]$env:MARKET_DATA_TOKEN -and ![bool]$env:TUSHARE_TOKEN) {
  Write-Warning 'Tushare token is not configured; API will use empty research data.'
}
Push-Location (Join-Path $root 'apps\api')
try { & (Join-Path $root '.venv\Scripts\python.exe') -m uvicorn core.app:app --host 127.0.0.1 --port $Port }
finally { Pop-Location }
