param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Manifest)) { throw "找不到发布清单: $Manifest" }
$release = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
if (!$release.git_sha -or !$release.version) { throw '发布清单缺少 git_sha 或 version' }
Write-Warning "将回滚到提交 $($release.git_sha)，请在目标编排环境执行该版本部署。"
Write-Output "目标版本: $($release.version)"
Write-Output "健康检查路径: $($release.health_path)"
