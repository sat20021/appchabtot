"""
Compatibility wrapper around the refactored ml_services.persona_service module.

New code should import from `ml_services.persona_service`. This shim exists for
older modules (and type checkers) that still expect `services.persona_service`.
"""

from ml_services.persona_service import (  # type: ignore F401
    Persona,
    PersonaService,
    get_persona_service,
)

__all__ = [
    "Persona",
    "PersonaService",
    "get_persona_service",
]

