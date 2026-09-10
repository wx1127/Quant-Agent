param(
  [string]$BaseUrl = "http://localhost:8000",
  [string]$Token = "health:admin"
)
$ErrorActionPreference = 'Stop'
$headers = @{ Authorization = "Bearer $Token" }
$health = Invoke-WebRequest -Uri "$BaseUrl/v1/health" -Headers $headers -Method Get
if ($health.StatusCode -ne 200) { throw "health endpoint returned $($health.StatusCode)" }
if (!$health.Headers['X-Request-ID']) { throw 'health response missing X-Request-ID' }
$metrics = Invoke-WebRequest -Uri "$BaseUrl/v1/metrics" -Headers $headers -Method Get
if ($metrics.StatusCode -ne 200) { throw "metrics endpoint returned $($metrics.StatusCode)" }
Write-Output "healthcheck passed: $BaseUrl"
