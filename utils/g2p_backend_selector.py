#!/usr/bin/env python3
"""
Dynamic G2P Backend Selector

Automatically detects available G2P backends and intelligently selects
the best one based on user preference and availability.

NO HARDCODED BACKEND SELECTION - fully dynamic detection and fallback.
"""

import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a G2P backend"""
    name: str
    available: bool
    priority: int  # Lower = higher priority
    description: str
    module_name: str = None
    
    def __repr__(self):
        status = "✅ Available" if self.available else "❌ Not installed"
        return f"{self.name}: {status} (priority: {self.priority})"


class G2PBackendSelector:
    """
    Dynamic G2P backend selector with automatic detection and fallback.
    
    Features:
    - Detects all available backends dynamically
    - Respects user preference
    - Automatically falls back to next available backend
    - No hardcoded backend selection
    - Provides clear feedback on selection process
    """
    
    # Backend definitions with priority (lower = better)
    # Priority is only used for automatic selection, NOT hardcoded
    BACKEND_DEFINITIONS = {
        'hybrid': {
            'module': None,  # Uses multiple backends intelligently
            'import_test': lambda: True,  # Always available (falls back gracefully)
            'priority': 0,  # BEST - combines all features
            'description': 'Hybrid - Enterprise G2P (epitran+rule_based+preprocessing)'
        },
        'transphone': {
            'module': 'transphone',
            'import_test': lambda: __import__('transphone'),
            'priority': 1,  # High quality single backend
            'description': 'Transphone - State-of-the-art multilingual G2P'
        },
        'epitran': {
            'module': 'epitran',
            'import_test': lambda: __import__('epitran'),
            'priority': 2,  # Good quality, well-tested
            'description': 'Epitran - Rule-based G2P with Ethiopic support'
        },
        'rule_based': {
            'module': None,  # Built-in, always available
            'import_test': lambda: True,
            'priority': 3,  # Fallback, always works
            'description': 'Rule-based - Built-in Amharic phonological rules'
        }
    }
    
    def __init__(self, verbose: bool = True):
        """
        Initialize backend selector
        
        Args:
            verbose: Print detection and selection messages
        """
        self.verbose = verbose
        self.available_backends = self._detect_available_backends()
        
        if self.verbose:
            self._print_detection_summary()
    
    def _detect_available_backends(self) -> Dict[str, BackendInfo]:
        """
        Detect which backends are actually available.
        
        Returns:
            Dictionary mapping backend name to BackendInfo
        """
        detected = {}
        
        for name, definition in self.BACKEND_DEFINITIONS.items():
            is_available = False
            
            try:
                # Test if backend can be imported
                import_test = definition['import_test']
                import_test()
                is_available = True
            except (ImportError, Exception):
                is_available = False
            
            detected[name] = BackendInfo(
                name=name,
                available=is_available,
                priority=definition['priority'],
                description=definition['description'],
                module_name=definition['module']
            )
        
        return detected
    
    def _print_detection_summary(self):
        """Print a summary of detected backends"""
        logger.info("=" * 70)
        logger.info("G2P Backend Detection")
        logger.info("=" * 70)
        
        for name, info in sorted(self.available_backends.items(), 
                                key=lambda x: x[1].priority):
            status = "✅" if info.available else "❌"
            logger.info(f"  {status} {info.name:12s} - {info.description}")
        
        logger.info("=" * 70)
    
    def get_available_backends(self) -> List[str]:
        """
        Get list of available backend names, sorted by priority.
        
        Returns:
            List of available backend names
        """
        available = [
            name for name, info in self.available_backends.items()
            if info.available
        ]
        
        # Sort by priority
        available.sort(key=lambda x: self.available_backends[x].priority)
        
        return available
    
    def select_backend(
        self, 
        preferred: Optional[str] = None,
        fallback: bool = True
    ) -> Tuple[str, str]:
        """
        Select the best available backend.
        
        Args:
            preferred: User's preferred backend (None for auto-select)
            fallback: Whether to fallback to next available if preferred unavailable
            
        Returns:
            Tuple of (selected_backend, reason)
        """
        available = self.get_available_backends()
        
        if not available:
            return None, "No G2P backends available (this should never happen)"
        
        # Case 1: No preference - use highest priority available
        if preferred is None:
            selected = available[0]  # Already sorted by priority
            reason = f"Auto-selected (highest priority available)"
            
            if self.verbose:
                logger.info(f"✅ Selected backend: {selected} ({reason})")
            
            return selected, reason
        
        # Normalize preferred backend name
        preferred = preferred.lower().replace('-', '_').replace(' ', '_')
        
        # Case 2: Preferred backend is available
        if preferred in self.available_backends and self.available_backends[preferred].available:
            reason = f"User preference (requested: {preferred})"
            
            if self.verbose:
                logger.info(f"✅ Selected backend: {preferred} ({reason})")
            
            return preferred, reason
        
        # Case 3: Preferred backend NOT available
        if not fallback:
            reason = f"Preferred '{preferred}' not available and fallback disabled"
            
            if self.verbose:
                logger.warning(f"⚠️ {reason}")
            
            return None, reason
        
        # Case 4: Fallback to next available
        selected = available[0]  # Highest priority available
        reason = f"Fallback (preferred '{preferred}' not available)"
        
        if self.verbose:
            logger.warning(f"⚠️ Preferred backend '{preferred}' not available")
            logger.info(f"✅ Falling back to: {selected}")
        
        return selected, reason
    
    def is_backend_available(self, backend: str) -> bool:
        """
        Check if a specific backend is available.
        
        Args:
            backend: Backend name to check
            
        Returns:
            True if backend is available
        """
        backend = backend.lower().replace('-', '_').replace(' ', '_')
        return (backend in self.available_backends and 
                self.available_backends[backend].available)
    
    def get_backend_info(self, backend: str) -> Optional[BackendInfo]:
        """
        Get information about a specific backend.
        
        Args:
            backend: Backend name
            
        Returns:
            BackendInfo or None if not found
        """
        backend = backend.lower().replace('-', '_').replace(' ', '_')
        return self.available_backends.get(backend)
    
    def get_recommendation(self) -> str:
        """
        Get the recommended backend (highest priority available).
        
        Returns:
            Recommended backend name
        """
        available = self.get_available_backends()
        return available[0] if available else 'rule_based'
    
    def print_status_report(self):
        """Print a detailed status report"""
        print("\n" + "=" * 70)
        print("📊 G2P BACKEND STATUS REPORT")
        print("=" * 70 + "\n")
        
        print("Available Backends (in priority order):")
        for backend in self.get_available_backends():
            info = self.available_backends[backend]
            print(f"  ✅ {info.name:12s} - {info.description}")
        
        print("\nUnavailable Backends:")
        unavailable = [name for name, info in self.available_backends.items() 
                      if not info.available]
        if unavailable:
            for name in unavailable:
                info = self.available_backends[name]
                print(f"  ❌ {info.name:12s} - {info.description}")
                if info.module_name:
                    print(f"     Install with: pip install {info.module_name}")
        else:
            print("  (None - all backends available!)")
        
        print("\nRecommendation:")
        recommended = self.get_recommendation()
        info = self.available_backends[recommended]
        print(f"  🎯 {recommended} - {info.description}")
        
        print("\n" + "=" * 70 + "\n")


# Module-level singleton for efficiency
_global_selector: Optional[G2PBackendSelector] = None


def get_g2p_backend_selector(force_refresh: bool = False) -> G2PBackendSelector:
    """
    Get the global G2P backend selector instance.
    
    Args:
        force_refresh: Force re-detection of backends
        
    Returns:
        G2PBackendSelector instance
    """
    global _global_selector
    
    if _global_selector is None or force_refresh:
        _global_selector = G2PBackendSelector(verbose=True)
    
    return _global_selector


def select_g2p_backend(
    preferred: Optional[str] = None,
    fallback: bool = True,
    verbose: bool = True
) -> Tuple[str, str]:
    """
    Convenience function to select a G2P backend.
    
    Args:
        preferred: User's preferred backend (None for auto)
        fallback: Enable automatic fallback
        verbose: Print selection messages
        
    Returns:
        Tuple of (selected_backend, reason)
    """
    selector = G2PBackendSelector(verbose=verbose)
    return selector.select_backend(preferred=preferred, fallback=fallback)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("G2P Backend Selector - Test")
    print("=" * 70)
    print()
    
    # Test 1: Auto-detection
    print("Test 1: Auto-detection")
    print("-" * 70)
    selector = G2PBackendSelector(verbose=True)
    selector.print_status_report()
    
    # Test 2: Select with preference
    print("\nTest 2: Backend Selection")
    print("-" * 70)
    
    test_preferences = [
        None,  # Auto-select
        'transphone',  # Preferred
        'epitran',  # Alternative
        'rule_based',  # Fallback
        'nonexistent',  # Invalid (should fallback)
    ]
    
    for pref in test_preferences:
        backend, reason = selector.select_backend(preferred=pref)
        print(f"\nPreference: {pref or 'Auto'}")
        print(f"  Selected:  {backend}")
        print(f"  Reason:    {reason}")
    
    # Test 3: Check availability
    print("\n\nTest 3: Availability Checks")
    print("-" * 70)
    for backend in ['transphone', 'epitran', 'rule_based', 'invalid']:
        available = selector.is_backend_available(backend)
        status = "✅ Available" if available else "❌ Not available"
        print(f"{backend:15s}: {status}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
