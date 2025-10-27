#!/usr/bin/env python3
"""
Validate YouTube cookies file and test authentication
Run this on Lightning AI to diagnose cookie issues
"""

import sys
from pathlib import Path
import http.cookiejar
from datetime import datetime
import yt_dlp

def validate_cookies_file(cookies_path):
    """Validate cookies file format and content"""
    print("=" * 60)
    print("YOUTUBE COOKIES VALIDATOR")
    print("=" * 60)
    
    # Check if file exists
    cookies_file = Path(cookies_path)
    if not cookies_file.exists():
        print(f"\n❌ ERROR: Cookies file not found at: {cookies_path}")
        return False
    
    print(f"\n✓ File exists: {cookies_path}")
    print(f"  Size: {cookies_file.stat().st_size} bytes")
    
    # Read first few lines
    try:
        with open(cookies_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            all_lines = f.readlines()
            
        print(f"\n📄 First line: {first_line}")
        print(f"📄 Total lines: {len(all_lines) + 1}")
        
        # Check format
        if first_line not in ['# HTTP Cookie File', '# Netscape HTTP Cookie File']:
            print(f"\n⚠️  WARNING: First line should be '# Netscape HTTP Cookie File'")
            print(f"   Current: '{first_line}'")
        else:
            print("\n✓ Correct Netscape format header")
            
        # Check for YouTube-specific cookies
        youtube_cookies = [line for line in all_lines if 'youtube.com' in line or '.google.com' in line]
        print(f"\n📊 Found {len(youtube_cookies)} YouTube/Google cookies")
        
        # Check for critical cookies
        critical_cookies = ['VISITOR_INFO1_LIVE', 'YSC', 'PREF', 'CONSENT']
        found_critical = []
        for cookie_name in critical_cookies:
            if any(cookie_name in line for line in all_lines):
                found_critical.append(cookie_name)
        
        print(f"\n🔑 Critical cookies found: {len(found_critical)}/{len(critical_cookies)}")
        for cookie in found_critical:
            print(f"   ✓ {cookie}")
        missing = set(critical_cookies) - set(found_critical)
        if missing:
            print(f"\n⚠️  Missing critical cookies:")
            for cookie in missing:
                print(f"   ✗ {cookie}")
                
    except Exception as e:
        print(f"\n❌ ERROR reading file: {e}")
        return False
    
    # Try to load with http.cookiejar
    print(f"\n🔍 Testing with http.cookiejar...")
    try:
        jar = http.cookiejar.MozillaCookieJar(cookies_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        print(f"✓ Successfully loaded {len(jar)} cookies")
        
        # Check for expired cookies
        now = datetime.now().timestamp()
        expired = sum(1 for cookie in jar if cookie.expires and cookie.expires < now)
        if expired:
            print(f"⚠️  WARNING: {expired} cookies are expired")
        else:
            print(f"✓ No expired cookies")
            
    except Exception as e:
        print(f"❌ ERROR loading with cookiejar: {e}")
        return False
    
    # Test with yt-dlp
    print(f"\n🧪 Testing with yt-dlp...")
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'cookiefile': cookies_path,
        'extract_flat': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)
            if info:
                print(f"✓ yt-dlp can authenticate with cookies")
                print(f"  Test video title: {info.get('title', 'N/A')}")
                return True
            else:
                print(f"❌ yt-dlp authentication failed")
                return False
    except Exception as e:
        error_msg = str(e)
        print(f"❌ yt-dlp test failed: {error_msg[:200]}")
        
        if "bot" in error_msg.lower() or "sign in" in error_msg.lower():
            print(f"\n⚠️  DIAGNOSIS: Cookies are INVALID or EXPIRED")
            print(f"\n🔧 SOLUTION:")
            print(f"   1. Open YouTube in your browser and log in")
            print(f"   2. Install 'Get cookies.txt LOCALLY' extension")
            print(f"   3. Export cookies while logged in")
            print(f"   4. Upload NEW cookies.txt to Lightning AI")
            print(f"   5. Verify cookies are from the SAME browser session")
        
        return False

if __name__ == "__main__":
    # Check common cookie locations
    possible_paths = [
        '/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt',
        'cookies.txt',
        sys.argv[1] if len(sys.argv) > 1 else None
    ]
    
    cookies_path = None
    for path in possible_paths:
        if path and Path(path).exists():
            cookies_path = path
            break
    
    if not cookies_path:
        print("❌ No cookies.txt file found!")
        print("\nSearched locations:")
        for path in possible_paths:
            if path:
                print(f"  - {path}")
        print("\nUsage: python validate_cookies.py [path/to/cookies.txt]")
        sys.exit(1)
    
    success = validate_cookies_file(cookies_path)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RESULT: Cookies are valid and working!")
    else:
        print("❌ RESULT: Cookies are INVALID - Please export fresh cookies")
        print("\n📝 Instructions:")
        print("   1. On your LOCAL computer (not Lightning AI):")
        print("   2. Open Chrome/Firefox")
        print("   3. Go to YouTube.com and LOG IN")
        print("   4. Install extension: 'Get cookies.txt LOCALLY'")
        print("   5. Click extension icon → Export cookies.txt")
        print("   6. Upload to Lightning AI at:")
        print("      /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/cookies.txt")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
