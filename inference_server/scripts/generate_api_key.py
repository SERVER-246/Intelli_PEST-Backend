#!/usr/bin/env python3
"""
API Key Generator Script
========================
Generate and manage API keys for the inference server.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_key(name: str, tier: str, description: str = "") -> dict:
    """Generate a new API key."""
    try:
        from inference_server.security import APIKeyManager
        
        manager = APIKeyManager()
        key_data = manager.generate_key(
            name=name,
            tier=tier,
            description=description,
        )
        
        return key_data
        
    except ImportError:
        # Fallback to simple key generation
        import secrets
        import hashlib
        
        raw_key = secrets.token_urlsafe(32)
        key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        
        return {
            "key": raw_key,
            "key_id": key_id,
            "name": name,
            "tier": tier,
            "created_at": datetime.utcnow().isoformat(),
        }


def list_keys() -> list:
    """List all API keys."""
    try:
        from inference_server.security import APIKeyManager
        
        manager = APIKeyManager()
        stats = manager.get_stats()
        
        return stats.get("keys", [])
        
    except ImportError:
        return []


def revoke_key(key_id: str) -> bool:
    """Revoke an API key."""
    try:
        from inference_server.security import APIKeyManager
        
        manager = APIKeyManager()
        return manager.revoke_key(key_id)
        
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="API Key Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate key:  python generate_api_key.py generate --name "my-app" --tier standard
  List keys:     python generate_api_key.py list
  Revoke key:    python generate_api_key.py revoke --key-id abc123
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new API key")
    gen_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Key name/identifier",
    )
    gen_parser.add_argument(
        "--tier",
        type=str,
        choices=["free", "standard", "premium", "admin"],
        default="free",
        help="Access tier (default: free)",
    )
    gen_parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Key description",
    )
    gen_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all API keys")
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke API key")
    revoke_parser.add_argument(
        "--key-id",
        type=str,
        required=True,
        help="Key ID to revoke",
    )
    
    args = parser.parse_args()
    
    if args.command == "generate":
        key_data = generate_key(args.name, args.tier, args.description)
        
        if args.json:
            print(json.dumps(key_data, indent=2))
        else:
            print("\n" + "=" * 60)
            print("NEW API KEY GENERATED")
            print("=" * 60)
            print(f"Key:         {key_data['key']}")
            print(f"Key ID:      {key_data['key_id']}")
            print(f"Name:        {key_data['name']}")
            print(f"Tier:        {key_data['tier']}")
            print(f"Created:     {key_data.get('created_at', 'N/A')}")
            print("=" * 60)
            print("\n⚠️  IMPORTANT: Save this key! It cannot be recovered.")
            print("\nUsage:")
            print("  Header:  X-API-Key: " + key_data['key'])
            print("  Query:   ?api_key=" + key_data['key'])
            print()
    
    elif args.command == "list":
        keys = list_keys()
        
        if args.json:
            print(json.dumps(keys, indent=2))
        else:
            if not keys:
                print("No API keys found.")
            else:
                print("\n" + "=" * 80)
                print(f"{'Key ID':<20} {'Name':<20} {'Tier':<10} {'Created':<25}")
                print("=" * 80)
                for key in keys:
                    print(f"{key.get('key_id', 'N/A'):<20} "
                          f"{key.get('name', 'N/A'):<20} "
                          f"{key.get('tier', 'N/A'):<10} "
                          f"{key.get('created_at', 'N/A'):<25}")
                print("=" * 80)
                print(f"Total: {len(keys)} keys")
                print()
    
    elif args.command == "revoke":
        success = revoke_key(args.key_id)
        
        if success:
            print(f"✓ Key {args.key_id} has been revoked.")
        else:
            print(f"✗ Failed to revoke key {args.key_id}.")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
