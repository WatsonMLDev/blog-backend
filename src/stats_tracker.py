import json
import os
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StatsTracker:
    """
    Tracks and persists chat statistics to a JSON file.
    Logs events like session creation and messages sent.
    """
    
    def __init__(self, stats_file: str = "./data/chat_stats.json"):
        """
        Initialize the stats tracker.
        
        Args:
            stats_file: Path to the JSON file for storing stats
        """
        self.stats_file = stats_file
        self._ensure_file_exists()
        logger.info(f"StatsTracker initialized with file: {stats_file}")
    
    def _ensure_file_exists(self):
        """Create the stats file and directory if they don't exist."""
        stats_path = Path(self.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not stats_path.exists():
            # Initialize with empty events list
            with open(self.stats_file, 'w') as f:
                json.dump({"events": []}, f)
            logger.info(f"Created new stats file: {self.stats_file}")
    
    def log_event(self, event_type: str, session_id: str, metadata: Dict[str, Any] = None):
        """
        Log an event to the stats file.
        
        Args:
            event_type: Type of event ('session_created', 'message_sent', etc.)
            session_id: The session ID associated with the event
            metadata: Additional metadata about the event
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
        try:
            # Read existing data
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
            
            # Append new event
            data["events"].append(event)
            
            # Write back
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Logged event: {event_type} for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def get_stats(self, days: int = None) -> Dict[str, Any]:
        """
        Get aggregated statistics from the log file.
        
        Args:
            days: If specified, only include events from the last N days
            
        Returns:
            Dictionary with various statistics
        """
        try:
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
            
            events = data.get("events", [])
            
            # Filter by date if specified
            if days is not None:
                cutoff_time = time.time() - (days * 24 * 60 * 60)
                events = [e for e in events if e.get("timestamp", 0) >= cutoff_time]
            
            # Calculate statistics
            stats = self._calculate_stats(events)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def _calculate_stats(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated statistics from events."""
        if not events:
            return {
                "total_events": 0,
                "total_sessions": 0,
                "total_messages": 0,
                "unique_sessions": 0,
                "first_event": None,
                "last_event": None,
                "events_by_type": {},
                "activity_by_day": {}
            }
        
        # Count events by type
        events_by_type = defaultdict(int)
        unique_sessions = set()
        activity_by_day = defaultdict(int)
        
        for event in events:
            event_type = event.get("event_type", "unknown")
            session_id = event.get("session_id")
            timestamp = event.get("timestamp")
            
            events_by_type[event_type] += 1
            
            if session_id:
                unique_sessions.add(session_id)
            
            # Group by day
            if timestamp:
                day = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                activity_by_day[day] += 1
        
        # Sort activity by day
        activity_by_day = dict(sorted(activity_by_day.items()))
        
        return {
            "total_events": len(events),
            "total_sessions": events_by_type.get("session_created", 0),
            "total_messages": events_by_type.get("message_sent", 0),
            "unique_sessions": len(unique_sessions),
            "first_event": events[0].get("datetime") if events else None,
            "last_event": events[-1].get("datetime") if events else None,
            "events_by_type": dict(events_by_type),
            "activity_by_day": activity_by_day,
            "recent_activity": {
                "last_24h": self._count_recent_events(events, hours=24),
                "last_7d": self._count_recent_events(events, hours=24*7),
                "last_30d": self._count_recent_events(events, hours=24*30),
            }
        }
    
    def _count_recent_events(self, events: List[Dict], hours: int) -> Dict[str, int]:
        """Count events within the specified number of hours."""
        cutoff_time = time.time() - (hours * 60 * 60)
        recent_events = [e for e in events if e.get("timestamp", 0) >= cutoff_time]
        
        events_by_type = defaultdict(int)
        unique_sessions = set()
        
        for event in recent_events:
            event_type = event.get("event_type", "unknown")
            session_id = event.get("session_id")
            
            events_by_type[event_type] += 1
            if session_id:
                unique_sessions.add(session_id)
        
        return {
            "total_events": len(recent_events),
            "sessions_created": events_by_type.get("session_created", 0),
            "messages_sent": events_by_type.get("message_sent", 0),
            "unique_sessions": len(unique_sessions)
        }
