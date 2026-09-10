param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Manifest)) { throw "找不到发布清单: $Manifest" }
$release = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
if (!$release.git_sha -or !$release.version) { throw '发布清单缺少 git_sha 或 version' }
Write-Warning "rollback target commit: $($release.git_sha)"
Write-Output "target version: $($release.version)"
Write-Output "health path: $($release.health_path)"
