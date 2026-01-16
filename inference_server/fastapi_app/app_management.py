"""
App Management Router
=====================
Handles app version checking, APK downloads, and model info.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Router
app_router = APIRouter(prefix="/api/app", tags=["App Management"])


# =============================================================================
# Configuration
# =============================================================================

# APK storage directory
APK_DIR = Path(r"D:\Intelli_PEST-Backend\Apk-versions")

# Model backup directory (for version info)
MODEL_BACKUP_DIR = Path(r"D:\Intelli_PEST-Backend\model_backups")

# Retrain status file (contains model version)
RETRAIN_STATUS_FILE = MODEL_BACKUP_DIR / "retrain_status.json"

# APK filename pattern: intelli-pest-release-YYYY-MM-DD_HHMMSS.apk
APK_PATTERN = re.compile(r"intelli-pest-release-(\d{4}-\d{2}-\d{2}_\d{6})\.apk")


# =============================================================================
# Response Models
# =============================================================================

class AppVersionResponse(BaseModel):
    """Response for app version check."""
    status: str
    latest_version: str
    latest_build_date: str
    download_url: str
    filename: str
    file_size_mb: float
    force_update: bool
    release_notes: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response for model info."""
    status: str
    model_version: str
    num_classes: int
    class_names: list
    last_trained: Optional[str] = None
    total_fine_tunes: int = 0
    total_comprehensive: int = 0


class AvailableApksResponse(BaseModel):
    """Response listing all available APKs."""
    status: str
    apks: list
    total: int


# =============================================================================
# Helper Functions
# =============================================================================

def parse_apk_timestamp(filename: str) -> Optional[datetime]:
    """Parse timestamp from APK filename."""
    match = APK_PATTERN.match(filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
        except ValueError:
            pass
    return None


def timestamp_to_version(dt: datetime) -> str:
    """Convert timestamp to semantic version."""
    # Format: YYYY.MMDD (e.g., 2026.0116) - matches Android app versionName format
    return f"{dt.year}.{dt.month:02d}{dt.day:02d}"


def get_latest_apk() -> Optional[dict]:
    """Get info about the latest APK."""
    if not APK_DIR.exists():
        logger.warning(f"APK directory does not exist: {APK_DIR}")
        return None
    
    apk_files = list(APK_DIR.glob("intelli-pest-release-*.apk"))
    if not apk_files:
        logger.warning(f"No APK files found in: {APK_DIR}")
        return None
    
    # Parse timestamps and find the latest
    latest_apk = None
    latest_timestamp = None
    
    for apk_path in apk_files:
        timestamp = parse_apk_timestamp(apk_path.name)
        if timestamp:
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_apk = apk_path
    
    if latest_apk is None:
        logger.warning("No valid APK files with proper naming found")
        return None
    
    # Get file size
    file_size = latest_apk.stat().st_size
    file_size_mb = round(file_size / (1024 * 1024), 2)
    
    # Generate version from timestamp
    version = timestamp_to_version(latest_timestamp)
    
    return {
        "filename": latest_apk.name,
        "path": latest_apk,
        "timestamp": latest_timestamp,
        "version": version,
        "file_size": file_size,
        "file_size_mb": file_size_mb,
    }


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Supports formats:
    - Semantic: "1.0.0", "1.2.3"
    - Date-based: "2026.0109.1430"
    
    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    # Normalize versions by replacing dashes with dots
    parts1 = v1.replace("-", ".").split(".")
    parts2 = v2.replace("-", ".").split(".")
    
    # Convert to integers for comparison
    nums1 = []
    nums2 = []
    
    for p in parts1:
        try:
            nums1.append(int(p))
        except ValueError:
            nums1.append(0)
    
    for p in parts2:
        try:
            nums2.append(int(p))
        except ValueError:
            nums2.append(0)
    
    # Pad shorter list with zeros
    max_len = max(len(nums1), len(nums2))
    nums1.extend([0] * (max_len - len(nums1)))
    nums2.extend([0] * (max_len - len(nums2)))
    
    # Compare element by element
    for n1, n2 in zip(nums1, nums2):
        if n1 < n2:
            return -1
        elif n1 > n2:
            return 1
    
    return 0


def get_model_info() -> dict:
    """Get current model information."""
    # Default values
    info = {
        "version": "1.0.0",
        "num_classes": 12,
        "class_names": [
            "Healthy",
            "Internode borer",
            "Pink borer",
            "Rat damage",
            "Stalk borer",
            "Top borer",
            "army worm",
            "mealy bug",
            "porcupine damage",
            "root borer",
            "termite",
            "junk",
        ],
        "last_trained": None,
        "total_fine_tunes": 0,
        "total_comprehensive": 0,
    }
    
    # Try to read from retrain status file
    if RETRAIN_STATUS_FILE.exists():
        try:
            with open(RETRAIN_STATUS_FILE, "r") as f:
                status = json.load(f)
            
            info["version"] = status.get("current_version_string", "v1.0.0")
            info["last_trained"] = status.get("last_trained")
            info["total_fine_tunes"] = status.get("total_fine_tunes", 0)
            info["total_comprehensive"] = status.get("total_comprehensive", 0)
            
        except Exception as e:
            logger.error(f"Failed to read retrain status: {e}")
    
    return info


# =============================================================================
# Endpoints
# =============================================================================

@app_router.get("/version", response_model=AppVersionResponse)
async def get_app_version(client_version: Optional[str] = None):
    """
    Get latest app version information.
    
    Returns the latest APK version, download URL, and whether update is forced.
    The app should call this on startup to check for updates.
    
    Args:
        client_version: The current version installed on the client (optional query param)
    """
    latest = get_latest_apk()
    
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NO_APK_AVAILABLE",
                "message": "No APK files available for download",
            }
        )
    
    # Build download URL (relative, will be accessed via same server)
    download_url = f"/api/app/download/{latest['filename']}"
    
    # Determine if force update is needed by comparing versions
    # Only force update if there's actually a newer version available
    force_update = False
    if client_version:
        # Compare client version with server's latest version
        # Any version older than server's latest will trigger force update
        force_update = compare_versions(client_version, latest["version"]) < 0
        logger.info(f"Version check: client={client_version}, server={latest['version']}, force_update={force_update}")
    else:
        # No client_version = legacy APK that doesn't send version
        logger.warning("No client_version provided - legacy APK detected, forcing update")
        force_update = True
    
    return AppVersionResponse(
        status="success",
        latest_version=latest["version"],
        latest_build_date=latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        download_url=download_url,
        filename=latest["filename"],
        file_size_mb=latest["file_size_mb"],
        force_update=force_update,
        release_notes=None,  # Could be loaded from a notes file if needed
    )


@app_router.get("/download/{filename}")
async def download_apk(filename: str):
    """
    Download an APK file.
    
    Args:
        filename: Name of the APK file to download
        
    Returns:
        The APK file for installation
    """
    # Validate filename to prevent path traversal
    if not APK_PATTERN.match(filename):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILENAME",
                "message": "Invalid APK filename format",
            }
        )
    
    apk_path = APK_DIR / filename
    
    if not apk_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "APK_NOT_FOUND",
                "message": f"APK file not found: {filename}",
            }
        )
    
    return FileResponse(
        path=str(apk_path),
        filename=filename,
        media_type="application/vnd.android.package-archive",
    )


@app_router.get("/list", response_model=AvailableApksResponse)
async def list_available_apks():
    """
    List all available APK versions.
    
    Returns a list of all APK files with their version info.
    """
    if not APK_DIR.exists():
        return AvailableApksResponse(
            status="success",
            apks=[],
            total=0,
        )
    
    apk_files = list(APK_DIR.glob("intelli-pest-release-*.apk"))
    apks = []
    
    for apk_path in apk_files:
        timestamp = parse_apk_timestamp(apk_path.name)
        if timestamp:
            file_size = apk_path.stat().st_size
            apks.append({
                "filename": apk_path.name,
                "version": timestamp_to_version(timestamp),
                "build_date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "download_url": f"/api/app/download/{apk_path.name}",
            })
    
    # Sort by timestamp (newest first)
    apks.sort(key=lambda x: x["build_date"], reverse=True)
    
    return AvailableApksResponse(
        status="success",
        apks=apks,
        total=len(apks),
    )


@app_router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info_endpoint():
    """
    Get current model information.
    
    Returns the model version, number of classes, and training info.
    The app can use this to display dynamic model information to users.
    """
    info = get_model_info()
    
    return ModelInfoResponse(
        status="success",
        model_version=info["version"],
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        last_trained=info["last_trained"],
        total_fine_tunes=info["total_fine_tunes"],
        total_comprehensive=info["total_comprehensive"],
    )
