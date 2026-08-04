param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\DevelopTool\MinConda\envs\Quant Agent\python.exe'
$tokenFile = 'D:\DevelopTool\Quant-Agent\secrets\market_data_token.txt'
$logDir = Join-Path $repo 'logs\p8-paper'
$outputRoot = Join-Path $repo 'data\paper\p8-t08'
$lockPath = Join-Path $logDir 'scheduler.lock'
$statusPath = Join-Path $logDir 'scheduler-status.json'
$logPath = Join-Path $logDir ("scheduler-{0}.log" -f (Get-Date -Format 'yyyy-MM-dd'))

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$lock = $null

function Write-SchedulerStatus {
    param(
        [string]$Status,
        [int]$ExitCode,
        [string]$Message
    )
    [ordered]@{
        schema_version = 'p8-paper-scheduler-v1'
        observed_at = (Get-Date).ToString('o')
        trading_date = (Get-Date -Format 'yyyy-MM-dd')
        status = $Status
        exit_code = $ExitCode
        message = $Message
        task_name = 'Quant-Agent-P8-Paper'
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )

    # The task starts at 16:00. Wait until the repository's point-in-time gate (16:05)
    # so a provider timestamp at exactly 16:00 cannot be mistaken for unavailable data.
    $safeTime = (Get-Date).Date.AddHours(16).AddMinutes(5)
    while ((Get-Date) -lt $safeTime) {
        Start-Sleep -Seconds 15
    }

    if (-not (Test-Path -LiteralPath $python)) {
        throw "Quant Agent Python environment not found"
    }
    if (-not (Test-Path -LiteralPath $tokenFile)) {
        throw "market data token file is unavailable"
    }

    $env:PYTHONPATH = "$repo\packages;$repo"
    $env:MARKET_DATA_TOKEN_FILE = $tokenFile
    $arguments = @(
        (Join-Path $repo 'scripts\paper\run_daily.py'),
        '--output-root', $outputRoot,
        '--source-cache', (Join-Path $repo 'data\shadow\shadow-20260701-historical-v1\snapshots'),
        '--source-cache', (Join-Path $repo 'data\research\current\snapshots')
    )
    $result = & $python @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $result | ForEach-Object { $_.ToString() } | Add-Content -LiteralPath $logPath -Encoding UTF8
    if ($exitCode -ne 0) {
        throw "P8 PAPER daily runner exited with code $exitCode"
    }
    Write-SchedulerStatus -Status 'SUCCEEDED' -ExitCode 0 -Message 'daily runner completed'
    exit 0
}
catch {
    $_.Exception.Message | Add-Content -LiteralPath $logPath -Encoding UTF8
    Write-SchedulerStatus -Status 'FAILED' -ExitCode 1 -Message $_.Exception.Message
    exit 1
}
finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
}
