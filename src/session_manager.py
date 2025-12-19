import uuid
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """Represents a chat session with history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class SessionManager:
    """Manages chat sessions with automatic TTL-based cleanup."""
    
    def __init__(self, session_ttl_minutes: int = 60):
        """
        Initialize the session manager.
        
        Args:
            session_ttl_minutes: Time in minutes before a session expires due to inactivity
        """
        self.sessions: Dict[str, ChatSession] = {}
        self.session_ttl_seconds = session_ttl_minutes * 60
        logger.info(f"SessionManager initialized with TTL of {session_ttl_minutes} minutes")
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            The newly created session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id=session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            The ChatSession if found and not expired, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        # Check if session has expired
        if self._is_session_expired(session):
            logger.info(f"Session expired: {session_id}")
            self.delete_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = time.time()
        return session
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a message to a session's chat history.
        
        Args:
            session_id: The session ID
            role: Either 'user' or 'assistant'
            content: The message content
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        message = ChatMessage(role=role, content=content)
        session.messages.append(message)
        session.last_activity = time.time()
        
        logger.debug(f"Added {role} message to session {session_id}")
        return True
    
    def get_history(self, session_id: str) -> Optional[List[Dict]]:
        """
        Get the chat history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of messages as dictionaries, or None if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in session.messages
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if self._is_session_expired(session)
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired session(s)")
        
        return len(expired_sessions)
    
    def _is_session_expired(self, session: ChatSession) -> bool:
        """
        Check if a session has expired based on last activity.
        
        Args:
            session: The session to check
            
        Returns:
            True if expired, False otherwise
        """
        return (time.time() - session.last_activity) > self.session_ttl_seconds
    
    def get_stats(self) -> Dict:
        """
        Get statistics about current sessions.
        
        Returns:
            Dictionary with session statistics
        """
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len([
                s for s in self.sessions.values()
                if not self._is_session_expired(s)
            ])
        }
