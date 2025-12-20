
import pytest
import time
from src.stats_tracker import StatsTracker
import os
import shutil

@pytest.fixture
def temp_stats_file(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "chat_stats.json"
    return str(p)

def test_calculate_stats(temp_stats_file):
    tracker = StatsTracker(stats_file=temp_stats_file)

    # Create some events
    now = time.time()
    events = []

    # Session 1 (Today)
    events.append({
        "event_type": "session_created",
        "session_id": "s1",
        "timestamp": now,
        "datetime": "2023-01-01"
    })
    events.append({
        "event_type": "message_sent",
        "session_id": "s1",
        "timestamp": now,
        "datetime": "2023-01-01"
    })
    events.append({
        "event_type": "inference_completed",
        "session_id": "s1",
        "timestamp": now,
        "metadata": {"latency": 1.0, "doc_count": 2, "intent": "search", "model": "m1"}
    })

    # Session 2 (3 days ago)
    t_3d = now - (3 * 24 * 3600)
    events.append({
        "event_type": "session_created",
        "session_id": "s2",
        "timestamp": t_3d,
        "datetime": "2022-12-29"
    })
    events.append({
        "event_type": "message_sent",
        "session_id": "s2",
        "timestamp": t_3d,
        "datetime": "2022-12-29"
    })

    # Session 3 (10 days ago)
    t_10d = now - (10 * 24 * 3600)
    events.append({
        "event_type": "session_created",
        "session_id": "s3",
        "timestamp": t_10d,
        "datetime": "2022-12-22"
    })

    stats = tracker._calculate_stats(events)

    assert stats["total_events"] == 6
    assert stats["total_sessions"] == 3
    assert stats["total_messages"] == 2
    assert stats["unique_sessions"] == 3

    # Performance metrics
    assert stats["performance_metrics"]["average_latency"] == 1.0
    assert stats["performance_metrics"]["average_docs_retrieved"] == 2.0

    # Recent activity - 24h (only s1)
    r24 = stats["recent_activity"]["last_24h"]
    assert r24["sessions_created"] == 1
    assert r24["messages_sent"] == 1
    assert r24["unique_sessions"] == 1

    # Recent activity - 7d (s1 + s2)
    r7d = stats["recent_activity"]["last_7d"]
    assert r7d["sessions_created"] == 2
    assert r7d["messages_sent"] == 2
    assert r7d["unique_sessions"] == 2

    # Recent activity - 30d (s1 + s2 + s3)
    r30d = stats["recent_activity"]["last_30d"]
    assert r30d["sessions_created"] == 3
    assert r30d["unique_sessions"] == 3
