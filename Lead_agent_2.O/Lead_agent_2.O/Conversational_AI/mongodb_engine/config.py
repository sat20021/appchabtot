"""
Configuration helpers for the MongoDB engine.

We intentionally keep this thin and delegate to the existing environment
variables already used by `database.mongodb.MongoDBHandler`.
"""

import os
from typing import Dict


def get_mongo_settings() -> Dict[str, str]:
    """
    Return normalized MongoDB settings.

    Uses the same env vars as the existing MongoDB handler to keep things
    consistent across the project.
    """

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB")

    # Collection names are centralized here so they can be changed in one place.
    job_listings_collection = os.getenv("MONGODB_JOB_LISTINGS_COLLECTION", "job_listings")

    return {
        "uri": uri,
        "db_name": db_name,
        "job_listings_collection": job_listings_collection,
    }




