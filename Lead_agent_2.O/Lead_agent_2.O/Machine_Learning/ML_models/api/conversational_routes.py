"""
Helpers to expose the original Conversational_AI FastAPI app inside the unified server.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, APIRouter
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)

LEGACY_CONV_PREFIX = "/legacy-conversation"

_LEGACY_APP: Optional[FastAPI] = None


def _load_legacy_app() -> FastAPI:
    """
    Import the legacy Conversational_AI FastAPI application lazily.

    Returns:
        FastAPI: The legacy application instance.
    """
    global _LEGACY_APP
    if _LEGACY_APP:
        return _LEGACY_APP

    try:
        ml_root = Path(__file__).resolve().parents[2]
        repo_root = ml_root.parent
        legacy_dir = repo_root / "Conversational_AI"
        
        logger.info(f"Looking for Conversational_AI at: {legacy_dir}")
        logger.info(f"ML root: {ml_root}")
        logger.info(f"Repo root: {repo_root}")

        if not legacy_dir.exists():
            raise FileNotFoundError(f"Conversational_AI directory not found at: {legacy_dir}")

        # Ensure both the project root and the Conversational_AI directory are importable,
        # so modules like `main` and `database` (which expect to be run from that folder)
        # can still be resolved when we import their API from the unified server process.
        paths_added = []
        for path in (legacy_dir, repo_root, ml_root):
            if path.exists():
                str_path = str(path)
                if str_path not in sys.path:
                    sys.path.append(str_path)
                    paths_added.append(str_path)
                    logger.info(f"Added '{str_path}' to sys.path for legacy API import")
            else:
                logger.warning(f"Path does not exist: {path}")

        logger.info("Attempting to import Conversational_AI.conv_api...")
        import importlib
        legacy_main_module = importlib.import_module("Conversational_AI.conv_main")
        logger.info("Imported Conversational_AI.conv_main successfully")

        original_main_module = sys.modules.get("main")
        sys.modules["main"] = legacy_main_module
        logger.info("Temporarily patched sys.modules['main'] for legacy import")
        try:
            from Conversational_AI.conv_api import app as conv_app  # type: ignore
        finally:
            if original_main_module is not None:
                sys.modules["main"] = original_main_module
                logger.info("Restored original sys.modules['main']")
            else:
                sys.modules.pop("main", None)
                logger.info("Removed temporary sys.modules['main']")
        
        _LEGACY_APP = conv_app
        logger.info(
            "✅ Legacy Conversational_AI application loaded with %d routes",
            len(conv_app.routes),
        )
        
        # Log route details for debugging
        for route in conv_app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                logger.debug(f"  Route: {list(route.methods)} {route.path} (include_in_schema={getattr(route, 'include_in_schema', True)})")
        
        return _LEGACY_APP
        
    except ImportError as e:
        logger.error(f"Failed to import Conversational_AI.conv_api: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Failed to load legacy app: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def include_legacy_conversation_routes(
    parent_app: FastAPI, prefix: str = LEGACY_CONV_PREFIX
) -> None:
    """
    Register the Conversational_AI FastAPI routes beneath the unified server so they
    appear in the primary OpenAPI schema.

    Args:
        parent_app: The main FastAPI application defined in `unified_server.py`.
        prefix: URL prefix under which the legacy API should be exposed.
    """
    try:
        legacy_app = _load_legacy_app()
        logger.info(f"Loaded legacy app with {len(legacy_app.routes)} total routes")
        
        # Use tag that will appear right after "Persona Management" alphabetically
        # "Persona Management" < "Legacy - Conversational" < "RAG"
        legacy_router = APIRouter(prefix=prefix, tags=["Legacy - Conversational"])
        
        routes_added = 0
        routes_skipped = 0
        
        for route in legacy_app.routes:
            if not isinstance(route, APIRoute):
                routes_skipped += 1
                logger.debug(f"Skipping non-APIRoute: {type(route)}")
                continue
            if not route.include_in_schema:
                routes_skipped += 1
                logger.debug(f"Skipping route with include_in_schema=False: {route.path}")
                continue

            # Build full path with prefix
            full_path = f"{prefix}{route.path}" if route.path != "/" else prefix
            if route.path == "/":
                full_path = prefix
            
            logger.debug(f"Adding route: {list(route.methods)} {full_path}")
            
            try:
                legacy_router.add_api_route(
                    path=route.path,
                    endpoint=route.endpoint,
                    methods=list(route.methods or []),
                    response_model=route.response_model,
                    status_code=route.status_code,
                    tags=route.tags or ["Legacy - Conversational"],
                    dependencies=route.dependencies,
                    summary=route.summary,
                    description=route.description,
                    response_description=route.response_description,
                    responses=route.responses,
                    deprecated=route.deprecated,
                    operation_id=route.operation_id or f"legacy_{route.path.replace('/', '_').replace('{', '').replace('}', '')}",
                    response_model_include=route.response_model_include,
                    response_model_exclude=route.response_model_exclude,
                    response_model_by_alias=route.response_model_by_alias,
                    response_model_exclude_unset=route.response_model_exclude_unset,
                    response_model_exclude_defaults=route.response_model_exclude_defaults,
                    response_model_exclude_none=route.response_model_exclude_none,
                    include_in_schema=True,  # Force include in schema
                    name=route.name,
                    callbacks=route.callbacks,
                    openapi_extra=route.openapi_extra,
                )
                routes_added += 1
            except Exception as route_err:
                logger.error(f"Failed to add route {route.path}: {route_err}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        logger.info(f"Added {routes_added} routes, skipped {routes_skipped} routes")
        
        if routes_added == 0:
            logger.warning("No routes were added to the legacy router!")
            return
        
        parent_app.include_router(legacy_router)
        logger.info(
            "✅ Included %d legacy Conversational_AI routes at '%s'",
            routes_added,
            prefix,
        )
        
        # Verify routes were added to parent app
        legacy_routes_in_app = [
            r for r in parent_app.routes 
            if hasattr(r, 'path') and prefix in getattr(r, 'path', '')
        ]
        logger.info(f"Verified: {len(legacy_routes_in_app)} routes with prefix '{prefix}' found in parent app")
        
    except Exception as e:
        logger.error(f"Failed to include legacy conversation routes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

