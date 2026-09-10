param([Parameter(Mandatory=$true)][string]$Version)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $status = git status --porcelain
  if ($status) { throw 'working tree is dirty; publish refused' }
  $sha = (git rev-parse HEAD).Trim()
  $manifestDir = Join-Path $PSScriptRoot 'releases'
  New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
  $manifest = [ordered]@{ version=$Version; git_sha=$sha; created_at=(Get-Date).ToUniversalTime().ToString("o"); health_path="/v1/health"; rollback_policy="immutable-commit" }
  $manifest | ConvertTo-Json | Set-Content (Join-Path $manifestDir "$Version.json") -Encoding utf8
  Write-Output "release manifest generated: $Version ($sha)"
} finally { Pop-Location }
