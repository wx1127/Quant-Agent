from core.alerts import AlertSeverity, evaluate_alerts


def test_alerts_prioritize_reconciliation_and_kill_switch() -> None:
    alerts = evaluate_alerts(
        requests=100,
        errors=10,
        average_seconds=2.0,
        reconciliation_failures=1,
        kill_switch_active=True,
    )
    assert [alert.severity for alert in alerts[:2]] == [AlertSeverity.P0, AlertSeverity.P0]
    assert {alert.code for alert in alerts} == {
        "RECONCILIATION_FAILURE",
        "KILL_SWITCH_ACTIVE",
        "API_ERROR_RATE",
        "API_LATENCY",
    }
