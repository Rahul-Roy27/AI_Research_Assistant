"""Configuration and settings for the AI Research Assistant.

This module defines application settings and configuration helpers.
"""

from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings for the AI Research Assistant."""
    app_title: str = "AI Research Assistant"
    default_source_path: str = "Data"
    max_results: int = 5


def load_settings() -> Settings:
    """Load application settings from environment or defaults."""
    return Settings()


settings = load_settings()
