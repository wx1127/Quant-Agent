param()

$ErrorActionPreference = 'Stop'
$taskName = 'Quant-Agent-P8-Paper'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$runner = Join-Path $repo 'scripts\paper\run_scheduled.ps1'
$powershell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`""

$action = New-ScheduledTaskAction `
    -Execute $powershell `
    -Argument $arguments `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At '16:00'
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Quant Agent P8-T08 PAPER validation. No live broker connection.' `
    -Force | Out-Null

$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
[ordered]@{
    task_name = $task.TaskName
    state = $task.State.ToString()
    next_run_time = $info.NextRunTime.ToString('o')
    start_when_available = $task.Settings.StartWhenAvailable
    allow_start_on_batteries = -not $task.Settings.DisallowStartIfOnBatteries
    stop_on_battery_change = $task.Settings.StopIfGoingOnBatteries
    multiple_instances = $task.Settings.MultipleInstances.ToString()
} | ConvertTo-Json
