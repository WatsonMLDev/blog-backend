import json
import time
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict
import logging
from pathlib import Path
import os

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
        self.events = []
        self.seen_events = set()
        self.last_file_pos = 0

        self._ensure_file_exists()

        # Initial load of stats to memory
        self._load_new_events()

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

    def _load_new_events(self):
        """
        Reads only new events from the stats file since the last read.
        Updates self.events and self.seen_events in place.
        """
        try:
            # Check if file exists and current size
            if not os.path.exists(self.stats_file):
                return

            # If file is smaller than last read position, it was likely truncated/rotated
            current_size = os.path.getsize(self.stats_file)
            if current_size < self.last_file_pos:
                logger.warning(f"Stats file truncated. Resetting stats. Old pos: {self.last_file_pos}, New size: {current_size}")
                self.last_file_pos = 0
                self.events = []
                self.seen_events = set()

            if current_size == self.last_file_pos:
                return # No new data

            with open(self.stats_file, 'r') as f:
                f.seek(self.last_file_pos)

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

                            if event_key not in self.seen_events:
                                self.seen_events.add(event_key)
                                self.events.append(event)

                        except json.JSONDecodeError:
                            continue

                self.last_file_pos = f.tell()

        except Exception as e:
            logger.error(f"Failed to load stats: {e}")

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
            
            # Optimistically update in-memory state to reflect this change immediately
            # Note: _load_new_events will re-read this from file later, but deduplication will handle it.
            # Actually, to avoid race conditions or double entry if file write succeeds but next read fails (rare),
            # we rely on _load_new_events to be the source of truth.
            # However, for immediate consistency if get_stats is called right after log_event without file flush:
            # Python's flush usually works fine.
            # We will just let _load_new_events pick it up.

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
            # Refresh events from file
            self._load_new_events()

            events = self.events
            
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
        
        # Initialize accumulators
        events_by_type = defaultdict(int)
        unique_sessions = set()
        activity_by_day = defaultdict(int)
        
        # Advanced statistics accumulators
        latencies = []
        doc_counts = []
        intent_counts = defaultdict(int)
        model_counts = defaultdict(int)

        # Recent activity accumulators
        now = time.time()
        # Cutoffs in seconds
        cutoff_24h = now - (24 * 3600)
        cutoff_7d = now - (7 * 24 * 3600)
        cutoff_30d = now - (30 * 24 * 3600)

        # Helper to init recent stats dict
        def init_recent_stats():
            return {
                "total_events": 0,
                "sessions_created": 0,
                "messages_sent": 0,
                "unique_sessions_set": set()
            }

        recent_stats = {
            "24h": init_recent_stats(),
            "7d": init_recent_stats(),
            "30d": init_recent_stats()
        }

        # Single pass iteration
        for event in events:
            event_type = event.get("event_type", "unknown")
            session_id = event.get("session_id")
            timestamp = event.get("timestamp", 0)
            
            # 1. Global counts
            events_by_type[event_type] += 1
            if session_id:
                unique_sessions.add(session_id)
            
            if timestamp:
                day = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                activity_by_day[day] += 1

            # 2. Inference stats
            if event_type == "inference_completed":
                metadata = event.get("metadata", {})
                if "latency" in metadata:
                    latencies.append(metadata["latency"])
                if "doc_count" in metadata:
                    doc_counts.append(metadata["doc_count"])
                if "intent" in metadata:
                    intent_counts[str(metadata["intent"])] += 1
                if "model" in metadata:
                    model_counts[str(metadata["model"])] += 1

            # 3. Recent activity stats
            # Check 30d first (most inclusive)
            if timestamp >= cutoff_30d:
                self._update_recent_stats(recent_stats["30d"], event_type, session_id)

                # Check 7d
                if timestamp >= cutoff_7d:
                    self._update_recent_stats(recent_stats["7d"], event_type, session_id)

                    # Check 24h
                    if timestamp >= cutoff_24h:
                        self._update_recent_stats(recent_stats["24h"], event_type, session_id)

        # Sort activity by day
        activity_by_day = dict(sorted(activity_by_day.items()))

        # Calculate averages/max
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        avg_docs = sum(doc_counts) / len(doc_counts) if doc_counts else 0

        # Engagement: avg messages per session
        total_sessions_created = events_by_type.get("session_created", 0)
        total_messages = events_by_type.get("message_sent", 0)
        avg_messages_per_session = total_messages / total_sessions_created if total_sessions_created > 0 else 0

        # Format recent activity output
        def format_recent(stats_dict):
            return {
                "total_events": stats_dict["total_events"],
                "sessions_created": stats_dict["sessions_created"],
                "messages_sent": stats_dict["messages_sent"],
                "unique_sessions": len(stats_dict["unique_sessions_set"])
            }

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
                "last_24h": format_recent(recent_stats["24h"]),
                "last_7d": format_recent(recent_stats["7d"]),
                "last_30d": format_recent(recent_stats["30d"]),
            },
            "performance_metrics": {
                "average_latency": round(avg_latency, 3),
                "max_latency": round(max_latency, 3),
                "average_docs_retrieved": round(avg_docs, 2)
            },
            "usage_insights": {
                "intent_distribution": dict(intent_counts),
                "model_usage": dict(model_counts)
            },
            "engagement_metrics": {
                "avg_messages_per_session": round(avg_messages_per_session, 2)
            }
        }

    def _update_recent_stats(self, stats, event_type, session_id):
        """Helper to update a recent stats accumulator."""
        stats["total_events"] += 1
        if event_type == "session_created":
            stats["sessions_created"] += 1
        elif event_type == "message_sent":
            stats["messages_sent"] += 1
        
        if session_id:
            stats["unique_sessions_set"].add(session_id)
