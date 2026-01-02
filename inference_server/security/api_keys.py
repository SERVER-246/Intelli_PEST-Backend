"""
API Key Management
==================
Secure API key generation, validation, and management.
"""

import hashlib
import secrets
import time
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class APIKeyTier(Enum):
    """API key tiers with different permissions."""
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class APIKeyInfo:
    """Information about an API key."""
    key_hash: str
    tier: APIKeyTier
    created_at: datetime
    expires_at: Optional[datetime] = None
    description: str = ""
    active: bool = True
    rate_limit: int = 100
    rate_window: int = 60
    batch_limit: int = 10
    features: List[str] = field(default_factory=lambda: ["predict"])
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if the key is valid."""
        if not self.active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_feature(self, feature: str) -> bool:
        """Check if key has access to a feature."""
        return "all" in self.features or feature in self.features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe for storage)."""
        return {
            "key_hash": self.key_hash,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "description": self.description,
            "active": self.active,
            "rate_limit": self.rate_limit,
            "rate_window": self.rate_window,
            "batch_limit": self.batch_limit,
            "features": self.features,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKeyInfo":
        """Create from dictionary."""
        return cls(
            key_hash=data["key_hash"],
            tier=APIKeyTier(data["tier"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            description=data.get("description", ""),
            active=data.get("active", True),
            rate_limit=data.get("rate_limit", 100),
            rate_window=data.get("rate_window", 60),
            batch_limit=data.get("batch_limit", 10),
            features=data.get("features", ["predict"]),
            usage_count=data.get("usage_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
        )


# Tier configurations
TIER_CONFIG = {
    APIKeyTier.FREE: {
        "rate_limit": 50,
        "rate_window": 60,
        "batch_limit": 5,
        "features": ["predict", "health", "classes"],
    },
    APIKeyTier.STANDARD: {
        "rate_limit": 200,
        "rate_window": 60,
        "batch_limit": 10,
        "features": ["predict", "batch", "health", "status", "classes"],
    },
    APIKeyTier.PREMIUM: {
        "rate_limit": 1000,
        "rate_window": 60,
        "batch_limit": 20,
        "features": ["predict", "batch", "health", "status", "classes", "models"],
    },
    APIKeyTier.ADMIN: {
        "rate_limit": -1,  # Unlimited
        "rate_window": 60,
        "batch_limit": 50,
        "features": ["all"],
    },
}


class APIKeyManager:
    """Manages API keys with secure storage."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize API key manager.
        
        Args:
            storage_path: Path to JSON file for key storage.
        """
        self.storage_path = storage_path
        self._keys: Dict[str, APIKeyInfo] = {}
        self._rate_limits: Dict[str, List[float]] = {}  # key_hash -> list of timestamps
        
        if self.storage_path and self.storage_path.exists():
            self._load_keys()
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _load_keys(self):
        """Load keys from storage."""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for key_hash, key_data in data.items():
                    self._keys[key_hash] = APIKeyInfo.from_dict(key_data)
            logger.info(f"Loaded {len(self._keys)} API keys from storage")
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
    
    def _save_keys(self):
        """Save keys to storage."""
        if not self.storage_path:
            return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {key_hash: key_info.to_dict() for key_hash, key_info in self._keys.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved API keys to storage")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def register_static_key(
        self,
        api_key: str,
        tier: str = "admin",
        name: str = "static_key",
        description: str = "Static API key",
    ) -> None:
        """
        Register a static/fixed API key (for testing or default access).
        
        Args:
            api_key: The exact API key string to register
            tier: API key tier (string)
            name: Name for the key
            description: Description for the key
        """
        # Handle tier as string
        if isinstance(tier, str):
            tier = APIKeyTier(tier)
        
        key_hash = self._hash_key(api_key)
        config = TIER_CONFIG[tier]
        
        key_info = APIKeyInfo(
            key_hash=key_hash,
            tier=tier,
            created_at=datetime.utcnow(),
            expires_at=None,  # Never expires
            description=description or name,
            rate_limit=config["rate_limit"],
            rate_window=config["rate_window"],
            batch_limit=config["batch_limit"],
            features=config["features"],
        )
        
        self._keys[key_hash] = key_info
        logger.info(f"Registered static {tier.value} API key: {name}")
    
    def generate_key(
        self,
        name: str = "",
        tier: APIKeyTier = None,
        description: str = "",
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a new API key.
        
        Args:
            name: Name for the key
            tier: API key tier (string or APIKeyTier)
            description: Description for the key
            expires_in_days: Days until expiration (None = never)
            
        Returns:
            Dictionary with key info (shown only once!)
        """
        # Handle tier as string or enum
        if tier is None:
            tier = APIKeyTier.FREE
        elif isinstance(tier, str):
            tier = APIKeyTier(tier)
        
        # Generate secure random key
        api_key = f"ip_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(api_key)
        
        # Get tier config
        config = TIER_CONFIG[tier]
        
        # Use name as description if description not provided
        if name and not description:
            description = name
        
        # Create key info
        key_info = APIKeyInfo(
            key_hash=key_hash,
            tier=tier,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            description=description,
            rate_limit=config["rate_limit"],
            rate_window=config["rate_window"],
            batch_limit=config["batch_limit"],
            features=config["features"],
        )
        
        self._keys[key_hash] = key_info
        self._save_keys()
        
        logger.info(f"Generated new {tier.value} API key: {key_hash[:16]}...")
        return {
            "key": api_key,
            "key_id": key_hash[:16],
            "name": name or description,
            "tier": tier.value,
            "created_at": key_info.created_at.isoformat(),
        }
    
    def validate_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Dictionary with validation result
        """
        if not api_key:
            return {"valid": False, "error": "No API key provided"}
            
        key_hash = self._hash_key(api_key)
        key_info = self._keys.get(key_hash)
        
        if not key_info:
            logger.warning(f"Invalid API key attempt: {key_hash[:16]}...")
            return {"valid": False, "error": "Invalid API key"}
        
        if not key_info.is_valid():
            logger.warning(f"Expired or inactive API key: {key_hash[:16]}...")
            return {"valid": False, "error": "Expired or inactive API key"}
        
        # Update usage stats
        key_info.usage_count += 1
        key_info.last_used = datetime.utcnow()
        self._save_keys()
        
        return {
            "valid": True,
            "tier": key_info.tier.value,
            "key_id": key_hash[:16],
        }
    
    def check_rate_limit(self, api_key: str) -> tuple:
        """
        Check if API key is within rate limit.
        
        Args:
            api_key: The API key to check
            
        Returns:
            Tuple of (allowed, info_dict)
        """
        key_hash = self._hash_key(api_key)
        key_info = self._keys.get(key_hash)
        
        if not key_info:
            return False, {"error": "Invalid API key"}
        
        # Unlimited rate limit
        if key_info.rate_limit == -1:
            return True, {"allowed": True}
        
        current_time = time.time()
        window_start = current_time - key_info.rate_window
        
        # Get or create rate limit tracking
        if key_hash not in self._rate_limits:
            self._rate_limits[key_hash] = []
        
        # Clean old entries
        self._rate_limits[key_hash] = [
            ts for ts in self._rate_limits[key_hash]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._rate_limits[key_hash]) >= key_info.rate_limit:
            oldest = min(self._rate_limits[key_hash])
            retry_after = int(oldest - window_start) + 1
            return False, {"retry_after": max(retry_after, 1)}
        
        # Add current request
        self._rate_limits[key_hash].append(current_time)
        return True, {"allowed": True}
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(api_key)
        if key_hash in self._keys:
            self._keys[key_hash].active = False
            self._save_keys()
            logger.info(f"Revoked API key: {key_hash[:16]}...")
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        return [
            {
                "key_preview": f"{k[:8]}...{k[-4:]}",
                **v.to_dict()
            }
            for k, v in self._keys.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about API keys."""
        return {
            "total_keys": len(self._keys),
            "active_keys": sum(1 for k in self._keys.values() if k.active),
            "by_tier": {
                tier.value: sum(1 for k in self._keys.values() if k.tier == tier)
                for tier in APIKeyTier
            },
            "keys": self.list_keys(),
        }
    
    def record_request(self, api_key: str):
        """Record a request for rate limiting."""
        allowed, _ = self.check_rate_limit(api_key)
        return allowed
    
    def add_admin_key(self, admin_key: str):
        """Add an admin key from environment."""
        if not admin_key or admin_key == "your_secure_admin_api_key_here":
            logger.warning("No admin API key configured!")
            return
        
        key_hash = self._hash_key(admin_key)
        if key_hash not in self._keys:
            config = TIER_CONFIG[APIKeyTier.ADMIN]
            self._keys[key_hash] = APIKeyInfo(
                key_hash=key_hash,
                tier=APIKeyTier.ADMIN,
                created_at=datetime.utcnow(),
                description="Admin key from environment",
                rate_limit=config["rate_limit"],
                rate_window=config["rate_window"],
                batch_limit=config["batch_limit"],
                features=config["features"],
            )
            logger.info("Admin API key configured from environment")


# Global manager instance
_manager: Optional[APIKeyManager] = None


def get_api_key_manager(storage_path: Optional[Path] = None) -> APIKeyManager:
    """Get or create the global API key manager."""
    global _manager
    if _manager is None:
        _manager = APIKeyManager(storage_path)
    return _manager


def verify_api_key(api_key: str) -> Optional[APIKeyInfo]:
    """Verify an API key and return its info."""
    manager = get_api_key_manager()
    return manager.validate_key(api_key)
