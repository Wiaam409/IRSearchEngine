from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

class IEventBus(ABC):
    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to an event type with a handler callback."""
        pass

    @abstractmethod
    def publish(self, event_type: str, payload: Dict[str, Any]):
        """Publish an event to all subscribers."""
        pass
