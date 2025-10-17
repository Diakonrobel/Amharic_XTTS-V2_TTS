"""
Proxy Manager for YouTube Downloads
Handles proxy rotation to bypass rate limits and regional restrictions.
"""

import random
import time
from typing import Optional, List
from pathlib import Path


class ProxyManager:
    """
    Manages a pool of proxies with rotation and health checking.
    """
    
    def __init__(self, proxy_list: Optional[List[str]] = None, proxy_file: Optional[str] = None):
        """
        Initialize proxy manager.
        
        Args:
            proxy_list: List of proxy URLs (e.g., ['http://user:pass@host:port', ...])
            proxy_file: Path to file containing proxy URLs (one per line)
        """
        self.proxies = []
        self.current_index = 0
        self.failed_proxies = set()
        
        if proxy_list:
            self.proxies = proxy_list
        elif proxy_file and Path(proxy_file).exists():
            self.load_from_file(proxy_file)
    
    def load_from_file(self, proxy_file: str):
        """
        Load proxies from a file (one proxy per line).
        
        Format:
            http://proxy1:port
            http://user:pass@proxy2:port
            socks5://proxy3:port
        """
        with open(proxy_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.proxies.append(line)
        print(f"Loaded {len(self.proxies)} proxies from {proxy_file}")
    
    def get_next_proxy(self) -> Optional[str]:
        """
        Get the next proxy in rotation, skipping failed ones.
        
        Returns:
            Proxy URL or None if no proxies available
        """
        if not self.proxies:
            return None
        
        # Filter out failed proxies
        available_proxies = [p for p in self.proxies if p not in self.failed_proxies]
        
        if not available_proxies:
            # Reset failed proxies if all have failed
            print("⚠ All proxies failed, resetting and trying again...")
            self.failed_proxies.clear()
            available_proxies = self.proxies
        
        # Get next proxy with rotation
        proxy = available_proxies[self.current_index % len(available_proxies)]
        self.current_index += 1
        
        return proxy
    
    def get_random_proxy(self) -> Optional[str]:
        """
        Get a random proxy from the pool.
        
        Returns:
            Random proxy URL or None
        """
        if not self.proxies:
            return None
        
        available_proxies = [p for p in self.proxies if p not in self.failed_proxies]
        if not available_proxies:
            self.failed_proxies.clear()
            available_proxies = self.proxies
        
        return random.choice(available_proxies)
    
    def mark_proxy_failed(self, proxy: str):
        """
        Mark a proxy as failed.
        
        Args:
            proxy: The proxy URL that failed
        """
        self.failed_proxies.add(proxy)
        print(f"⚠ Marked proxy as failed: {proxy}")
    
    def mark_proxy_working(self, proxy: str):
        """
        Mark a proxy as working (remove from failed set).
        
        Args:
            proxy: The proxy URL that's working
        """
        if proxy in self.failed_proxies:
            self.failed_proxies.remove(proxy)
            print(f"✓ Proxy is working again: {proxy}")


def create_proxy_manager_from_env() -> Optional[ProxyManager]:
    """
    Create a ProxyManager from environment variables.
    
    Environment variables:
        YTDLP_PROXY: Single proxy URL
        YTDLP_PROXY_LIST: Comma-separated list of proxy URLs
        YTDLP_PROXY_FILE: Path to file with proxy list
    
    Returns:
        ProxyManager instance or None
    """
    import os
    
    # Check for proxy list file
    proxy_file = os.getenv("YTDLP_PROXY_FILE")
    if proxy_file and Path(proxy_file).exists():
        return ProxyManager(proxy_file=proxy_file)
    
    # Check for comma-separated proxy list
    proxy_list_str = os.getenv("YTDLP_PROXY_LIST")
    if proxy_list_str:
        proxy_list = [p.strip() for p in proxy_list_str.split(',') if p.strip()]
        if proxy_list:
            return ProxyManager(proxy_list=proxy_list)
    
    # Check for single proxy
    single_proxy = os.getenv("YTDLP_PROXY")
    if single_proxy:
        return ProxyManager(proxy_list=[single_proxy])
    
    return None


# Example proxy lists for reference (these are public and may not work)
EXAMPLE_FREE_PROXY_SOURCES = [
    "https://www.proxy-list.download/api/v1/get?type=http",
    "https://api.proxyscrape.com/v2/?request=get&protocol=http",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
]

# Note: Free proxies are often unreliable. For production use, consider:
# - Paid proxy services (Bright Data, Oxylabs, Smartproxy, etc.)
# - Residential proxies for better success rates
# - Rotating proxy services with automatic failover
