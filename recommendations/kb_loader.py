#!/usr/bin/env python3
"""
kb_loader.py

PostgreSQL Knowledge Base Loader

Loads KB entries from JSON files and provides:
- Single file loading
- Directory loading
- Validation of entries
- Caching for performance

Usage:
    from kb_loader import KBLoader
    
    # Load from single file
    loader = KBLoader()
    kb = loader.load_file("data/kb_unified.json")
    
    # Load from directory
    kb = loader.load_directory("data/")
    
    # Get all entries
    for entry in kb.entries:
        print(entry.metadata.kb_id, entry.metadata.category)
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .kb_schema import KBEntry, KBVersion, Metadata, ProblemIdentity


logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a load operation."""
    success: bool
    entry_count: int
    entries: List[KBEntry]
    errors: List[str]
    source: str
    
    def __init__(self, source: str):
        self.success = True
        self.entry_count = 0
        self.entries = []
        self.errors = []
        self.source = source


class KBLoader:
    """
    Knowledge Base Loader.
    
    Loads KB entries from JSON files with validation.
    Supports single files, multiple files, and directories.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the KB loader.
        
        Args:
            cache_enabled: Whether to cache loaded entries
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, KBVersion] = {}
    
    def load_file(self, filepath: str) -> KBVersion:
        """
        Load KB from a single JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            KBVersion object with loaded entries
        """
        # Check cache
        if self.cache_enabled and filepath in self._cache:
            logger.info(f"Loading from cache: {filepath}")
            return self._cache[filepath]
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"KB file not found: {filepath}")
        
        logger.info(f"Loading KB from: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kb = KBVersion.from_dict(data)
        
        # Validate entries
        valid_entries = []
        for entry in kb.entries:
            if self._validate_entry(entry):
                valid_entries.append(entry)
            else:
                logger.warning(f"Skipping invalid entry: {entry.metadata.kb_id}")
        
        kb.entries = valid_entries
        kb.entry_count = len(valid_entries)
        
        # Cache if enabled
        if self.cache_enabled:
            self._cache[filepath] = kb
        
        logger.info(f"Loaded {kb.entry_count} valid entries")
        
        return kb
    
    def load_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        recursive: bool = True
    ) -> KBVersion:
        """
        Load all KB files from a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            Combined KBVersion with all entries
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        logger.info(f"Loading KB from directory: {directory}")
        
        # Find all JSON files
        if recursive:
            json_files = list(directory.rglob(pattern))
        else:
            json_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Load each file
        all_entries = []
        errors = []
        
        for filepath in sorted(json_files):
            try:
                kb = self.load_file(str(filepath))
                all_entries.extend(kb.entries)
                logger.debug(f"Loaded {len(kb.entries)} from {filepath.name}")
            except Exception as e:
                error = f"Error loading {filepath}: {str(e)}"
                logger.warning(error)
                errors.append(error)
        
        # Create combined KB
        combined_kb = KBVersion(
            kb_version="1.0.0",
            entry_count=len(all_entries),
            entries=all_entries,
        )
        
        if errors:
            logger.warning(f"Completed with {len(errors)} errors")
        
        logger.info(f"Total entries loaded: {len(all_entries)}")
        
        return combined_kb
    
    def load_multiple_files(self, filepaths: List[str]) -> KBVersion:
        """
        Load KB from multiple specific files.
        
        Args:
            filepaths: List of file paths
            
        Returns:
            Combined KBVersion with all entries
        """
        all_entries = []
        
        for filepath in filepaths:
            try:
                kb = self.load_file(filepath)
                all_entries.extend(kb.entries)
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
        
        return KBVersion(
            kb_version="1.0.0",
            entry_count=len(all_entries),
            entries=all_entries,
        )
    
    def _validate_entry(self, entry: KBEntry) -> bool:
        """
        Validate a single KB entry.
        
        Args:
            entry: The entry to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not entry.metadata.kb_id:
            logger.warning("Entry missing kb_id")
            return False
        
        if not entry.metadata.category:
            logger.warning(f"Entry {entry.metadata.kb_id} missing category")
            return False
        
        if not entry.problem_identity.issue_type:
            logger.warning(f"Entry {entry.metadata.kb_id} missing issue_type")
            return False
        
        if not entry.root_cause_analysis.primary_cause:
            logger.warning(f"Entry {entry.metadata.kb_id} missing primary_cause")
            return False
        
        # Check confidence score is valid
        if entry.confidence.confidence_score < 0 or entry.confidence.confidence_score > 1:
            logger.warning(f"Entry {entry.metadata.kb_id} has invalid confidence score")
            return False
        
        return True
    
    def get_entries_by_category(
        self,
        kb: KBVersion,
        category: str
    ) -> List[KBEntry]:
        """
        Filter entries by category.
        
        Args:
            kb: The knowledge base
            category: Category to filter by
            
        Returns:
            List of matching entries
        """
        return [e for e in kb.entries if e.matches_category(category)]
    
    def get_entries_by_severity(
        self,
        kb: KBVersion,
        severity: str
    ) -> List[KBEntry]:
        """
        Filter entries by severity.
        
        Args:
            kb: The knowledge base
            severity: Severity level to filter by
            
        Returns:
            List of matching entries
        """
        return [e for e in kb.entries if e.matches_severity(severity)]
    
    def get_entries_by_table(
        self,
        kb: KBVersion,
        table: str
    ) -> List[KBEntry]:
        """
        Filter entries that mention a specific table.
        
        Args:
            kb: The knowledge base
            table: Table name to search for
            
        Returns:
            List of matching entries
        """
        return [e for e in kb.entries if e.matches_table(table)]
    
    def get_entries_by_symptom(
        self,
        kb: KBVersion,
        symptom: str
    ) -> List[KBEntry]:
        """
        Filter entries that mention a symptom.
        
        Args:
            kb: The knowledge base
            symptom: Symptom to search for
            
        Returns:
            List of matching entries
        """
        return [e for e in kb.entries if e.matches_symptom(symptom)]
    
    def get_entries_by_cause(
        self,
        kb: KBVersion,
        cause: str
    ) -> List[KBEntry]:
        """
        Filter entries that mention a cause.
        
        Args:
            kb: The knowledge base
            cause: Cause to search for
            
        Returns:
            List of matching entries
        """
        return [e for e in kb.entries if e.matches_cause(cause)]
    
    def get_all_categories(self, kb: KBVersion) -> List[str]:
        """Get all unique categories in the KB."""
        categories = set()
        for entry in kb.entries:
            categories.add(entry.metadata.category)
        return sorted(categories)
    
    def get_all_severities(self, kb: KBVersion) -> List[str]:
        """Get all unique severity levels in the KB."""
        severities = set()
        for entry in kb.entries:
            severities.add(entry.metadata.severity)
        return sorted(severities)
    
    def get_statistics(self, kb: KBVersion) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Args:
            kb: The knowledge base
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_entries": len(kb.entries),
            "categories": {},
            "severities": {},
            "avg_confidence": 0,
        }
        
        # Count by category
        for entry in kb.entries:
            cat = entry.metadata.category
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            
            sev = entry.metadata.severity
            stats["severities"][sev] = stats["severities"].get(sev, 0) + 1
        
        # Calculate average confidence
        if kb.entries:
            total_conf = sum(e.confidence.confidence_score for e in kb.entries)
            stats["avg_confidence"] = total_conf / len(kb.entries)
        
        # Count entries with SQL examples
        sql_count = sum(1 for e in kb.entries if e.get_sql_examples())
        stats["entries_with_sql"] = sql_count
        
        # Count entries with config examples
        config_count = sum(1 for e in kb.entries if e.get_config_examples())
        stats["entries_with_config"] = config_count
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the loaded cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def preload(self, directory: str = "data/") -> None:
        """
        Preload all KB files from a directory.
        
        Args:
            directory: Directory to search for KB files
        """
        logger.info(f"Preloading KB from: {directory}")
        try:
            kb = self.load_directory(directory)
            logger.info(f"Preloaded {kb.entry_count} entries")
        except Exception as e:
            logger.error(f"Error preloading KB: {e}")


class KBFileWatcher:
    """
    Watches KB files for changes and reloads automatically.
    
    Useful for development when KB files are being updated.
    """
    
    def __init__(
        self,
        loader: KBLoader,
        check_interval: float = 5.0
    ):
        """
        Initialize the file watcher.
        
        Args:
            loader: KBLoader instance
            check_interval: How often to check for changes (seconds)
        """
        self.loader = loader
        self.check_interval = check_interval
        self._file_mtimes: Dict[str, float] = {}
        self._running = False
    
    def _get_mtime(self, filepath: str) -> float:
        """Get modification time of a file."""
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return 0
    
    def _scan_files(self, directory: str) -> List[str]:
        """Get all KB files in a directory."""
        path = Path(directory)
        if path.is_file():
            return [str(path)]
        return list(path.rglob("*.json"))
    
    def start(self, directory: str = "data/") -> None:
        """Start watching for file changes."""
        if self._running:
            logger.warning("File watcher already running")
            return
        
        self._running = True
        files = self._scan_files(directory)
        
        for filepath in files:
            self._file_mtimes[filepath] = self._get_mtime(filepath)
        
        logger.info(f"Watching {len(files)} files for changes")
    
    def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        logger.info("File watcher stopped")
    
    def check_changes(self, directory: str = "data/") -> bool:
        """
        Check if any files have changed.
        
        Args:
            directory: Directory to check
            
        Returns:
            True if changes detected, False otherwise
        """
        if not self._running:
            return False
        
        files = self._scan_files(directory)
        
        for filepath in files:
            current_mtime = self._get_mtime(filepath)
            old_mtime = self._file_mtimes.get(filepath, 0)
            
            if current_mtime > old_mtime:
                logger.info(f"File changed: {filepath}")
                self._file_mtimes[filepath] = current_mtime
                return True
        
        return False


def load_kb_from_default_locations() -> Optional[KBVersion]:
    """
    Try to load KB from default locations.
    
    Checks:
    1. data/kb_unified.json
    2. data/kb_sample.json
    3. data/kb_schema.json (if it contains entries)
    
    Returns:
        Loaded KBVersion or None if not found
    """
    loader = KBLoader()
    
    # Try common locations
    locations = [
        "data/kb_unified.json",
        "data/kb_sample.json",
        "data/kb_schema.json",
    ]
    
    for location in locations:
        if os.path.exists(location):
            try:
                kb = loader.load_file(location)
                if kb.entry_count > 0:
                    logger.info(f"Loaded KB from {location}")
                    return kb
            except Exception as e:
                logger.debug(f"Could not load from {location}: {e}")
    
    # Try loading from data directory
    if os.path.exists("data/"):
        try:
            kb = loader.load_directory("data/")
            if kb.entry_count > 0:
                logger.info(f"Loaded {kb.entry_count} entries from data/ directory")
                return kb
        except Exception as e:
            logger.debug(f"Could not load from data/ directory: {e}")
    
    logger.warning("No KB files found in default locations")
    return None

