"""
Compatibility shim for legacy imports.

Historically, service modules lived under `services.*`. The new structure
renamed the package to `ml_services.*`, but some callers (especially in the
Conversational_AI package) still import from the old path. To keep backward
compatibility—and to keep static analyzers happy—we provide lightweight
re-export modules here that proxy to the new implementations.
"""

# Nothing to export at the package level yet, but the module exists so that
# `import services.*` resolves cleanly.

