"""GA session manager for handling multiple concurrent sessions."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from .genetic_algorithm import GeneticAlgorithm, GAConfig


class GASession:
    """Represents a single GA session."""

    def __init__(self, session_id: str, ga: GeneticAlgorithm):
        """
        Initialize a GA session.
        
        Args:
            session_id: Unique session identifier
            ga: GeneticAlgorithm instance
        """
        self.session_id = session_id
        self.ga = ga
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.is_active = False

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired."""
        return (datetime.now() - self.last_accessed) > timedelta(minutes=timeout_minutes)

    def update_accessed(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now()


class GASessionManager:
    """Manages multiple GA sessions."""

    def __init__(self, max_sessions: int = 100, timeout_minutes: int = 30):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
            timeout_minutes: Session timeout in minutes
        """
        self.sessions: Dict[str, GASession] = {}
        self.max_sessions = max_sessions
        self.timeout_minutes = timeout_minutes

    def create_session(self, config: Optional[GAConfig] = None) -> str:
        """
        Create a new GA session.
        
        Args:
            config: Optional GAConfig object
            
        Returns:
            Session ID
        """
        # Clean expired sessions
        self._cleanup_expired_sessions()

        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self.max_sessions}) reached")

        session_id = str(uuid.uuid4())
        ga_config = config or GAConfig()
        ga = GeneticAlgorithm(ga_config)

        session = GASession(session_id, ga)
        self.sessions[session_id] = session

        return session_id

    def get_session(self, session_id: str) -> Optional[GASession]:
        """Get a session by ID."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        session.update_accessed()
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.timeout_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]

    def get_sessions_count(self) -> int:
        """Get number of active sessions."""
        self._cleanup_expired_sessions()
        return len(self.sessions)


# Global session manager
_session_manager: Optional[GASessionManager] = None


def get_session_manager() -> GASessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = GASessionManager()
    return _session_manager
