"""
Preprocessing Module: Data collection and parsing pipeline.

Modules:
    - scraper: IMSDb movie script scraper (all-or-nothing strategy)
    - parser: Script text parser (extracts scenes, characters, interactions)
"""

from .scraper import get_movie_list, scrape_detail_page, scrape_script, main as scrape_main
from .parser import parse_script, parse_single_file, main as parse_main

__all__ = [
    # Scraper
    "get_movie_list",
    "scrape_detail_page",
    "scrape_script",
    "scrape_main",
    # Parser
    "parse_script",
    "parse_single_file",
    "parse_main",
]
