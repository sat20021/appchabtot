"""
Lightweight MongoDB client helpers for job-related collections.

These helpers intentionally mirror the existing `database.mongodb.MongoDBHandler`
configuration so that all MongoDB access stays consistent.
"""

from typing import Any

from pymongo import MongoClient

from mongodb_engine.config import get_mongo_settings


def get_mongo_client() -> MongoClient:
    """Create a new MongoClient using shared settings."""

    settings = get_mongo_settings()
    if not settings["uri"]:
        raise RuntimeError("MONGODB_URI is not configured")

    return MongoClient(settings["uri"])


def get_job_listings_collection() -> Any:
    """
    Return a handle to the `job_listings` collection.

    NOTE: Callers are responsible for closing the underlying client if they
    create long‑lived objects. For short lived queries (the common case here),
    it's acceptable to rely on process shutdown to clean up connections.
    """

    settings = get_mongo_settings()
    if not settings["db_name"]:
        raise RuntimeError("MONGODB_DB is not configured")

    client = get_mongo_client()
    db = client[settings["db_name"]]
    return db[settings["job_listings_collection"]]




