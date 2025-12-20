import json
import time
from typing import Dict, List, Any
from datetime import datetime
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
        """Create the stats file and directory if they don't exist. handles migration from legacy JSON."""
        stats_path = Path(self.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not stats_path.exists():
            # Create empty file
            stats_path.touch()
            logger.info(f"Created new stats file: {self.stats_file}")
        else:
            # Check for legacy format and migrate if needed
            self._migrate_if_needed()

    def _migrate_if_needed(self):
        """Migrates legacy JSON format to JSONL if detected."""
        try:
            with open(self.stats_file, 'r') as f:
                # Read a small chunk to detect format, skipping whitespace
                start_content = f.read(1024).strip()
                if not start_content or not start_content.startswith('{'):
                    return # Not JSON or empty

                # It might be legacy JSON or JSONL.
                # Reset file pointer
                f.seek(0)

                try:
                    # Try to load as a single JSON object (Legacy format)
                    data = json.load(f)

                    # Verify it's actually the legacy structure
                    if not isinstance(data, dict) or "events" not in data:
                        return # Not legacy format (might be single line JSONL)

                    events = data.get("events", [])
                except json.JSONDecodeError:
                    # If JSONDecodeError, it likely means multiple JSON objects (JSONL)
                    # or invalid JSON. In either case, we don't migrate.
                    return

            # Write back in JSONL format
            logger.info(f"Migrating stats file {self.stats_file} to JSONL format...")
            with open(self.stats_file, 'w') as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")
            logger.info("Migration complete.")

        except Exception as e:
            logger.error(f"Error checking/migrating stats file: {e}")

    def log_event(self, event_type: str, session_id: str, metadata: Dict[str, Any] = None):
        """
        Log an event to the stats file (append-only JSONL).
        
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
            # Append new event as a single line
            with open(self.stats_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
            
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
            events = []
            seen_events = set()

            with open(self.stats_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)

                            # Deduplicate events based on content
                            # Using tuple of (timestamp, event_type, session_id, sorted metadata items)
                            metadata = event.get("metadata", {})
                            metadata_key = tuple(sorted((str(k), str(v)) for k, v in metadata.items())) if metadata else ()

                            event_key = (
                                event.get("timestamp"),
                                event.get("event_type"),
                                event.get("session_id"),
                                metadata_key
                            )

                            if event_key not in seen_events:
                                seen_events.add(event_key)
                                events.append(event)

                        except json.JSONDecodeError:
                            continue
            
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
