class TrackingManager:
    """
    Global registry for tracking identities.
    Ensures absolute uniqueness of Instance IDs across the entire synthetic timeline.
    """
    _current_id: int = 0

    @classmethod
    def get_next_id(cls) -> int:
        """Allocates and returns a universally unique track ID."""
        cls._current_id += 1
        return cls._current_id

    @classmethod
    def reset(cls) -> None:
        """Resets the ID counter (typically invoked when starting a new batch generation)."""
        cls._current_id = 0