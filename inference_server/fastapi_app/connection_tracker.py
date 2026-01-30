"""
Connection Quality Tracker
==========================
Stores and analyzes connection quality data from mobile app users.
Tracks network types, speeds, and locations to understand connectivity patterns.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConnectionQualityTracker:
    """
    Tracks connection quality data reported by the mobile app.
    
    Features:
    - Stores connection samples with location data
    - Analyzes slow connection patterns by location
    - Generates statistics for network quality mapping
    """
    
    QUALITY_LEVELS = {
        0: "offline",
        1: "very_slow",
        2: "slow",
        3: "moderate", 
        4: "good",
        5: "excellent"
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the connection tracker.
        
        Args:
            data_dir: Directory to store the SQLite database
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "connection_data")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "connection_quality.db"
        self._lock = threading.Lock()
        
        self._init_database()
        logger.info(f"ConnectionQualityTracker initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Connection samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    user_id TEXT,
                    app_version TEXT,
                    timestamp INTEGER NOT NULL,
                    network_type TEXT NOT NULL,
                    quality_level INTEGER NOT NULL,
                    download_speed_kbps INTEGER,
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_timestamp 
                ON connection_samples(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_network_type 
                ON connection_samples(network_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_quality 
                ON connection_samples(quality_level)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_location 
                ON connection_samples(latitude, longitude)
            """)
            
            # Aggregated location stats table (for faster queries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lat_bucket REAL NOT NULL,
                    lng_bucket REAL NOT NULL,
                    network_type TEXT NOT NULL,
                    avg_quality REAL,
                    avg_speed_kbps REAL,
                    sample_count INTEGER DEFAULT 0,
                    slow_sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lat_bucket, lng_bucket, network_type)
                )
            """)
            
            conn.commit()
    
    def record_samples(
        self,
        device_id: str,
        user_id: Optional[str],
        app_version: str,
        samples: List[Dict[str, Any]]
    ) -> int:
        """Record connection samples from the app.
        
        Args:
            device_id: Device identifier
            user_id: Optional user identifier
            app_version: App version string
            samples: List of connection samples
            
        Returns:
            Number of samples recorded
        """
        if not samples:
            return 0
        
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    
                    recorded = 0
                    for sample in samples:
                        try:
                            cursor.execute("""
                                INSERT INTO connection_samples 
                                (device_id, user_id, app_version, timestamp, network_type, 
                                 quality_level, download_speed_kbps, latitude, longitude)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                device_id,
                                user_id,
                                app_version,
                                sample.get("timestamp", 0),
                                sample.get("network_type", "unknown"),
                                sample.get("quality_level", 3),
                                sample.get("download_speed_kbps"),
                                sample.get("latitude"),
                                sample.get("longitude")
                            ))
                            recorded += 1
                            
                            # Update location stats if location is available
                            lat = sample.get("latitude")
                            lng = sample.get("longitude")
                            if lat is not None and lng is not None:
                                self._update_location_stats(
                                    cursor, lat, lng,
                                    sample.get("network_type", "unknown"),
                                    sample.get("quality_level", 3),
                                    sample.get("download_speed_kbps")
                                )
                        except Exception as e:
                            logger.warning(f"Failed to record sample: {e}")
                    
                    conn.commit()
                    logger.info(f"Recorded {recorded} connection samples from device {device_id}")
                    return recorded
                    
            except Exception as e:
                logger.error(f"Failed to record connection samples: {e}")
                return 0
    
    def _update_location_stats(
        self,
        cursor,
        latitude: float,
        longitude: float,
        network_type: str,
        quality_level: int,
        speed_kbps: Optional[int]
    ):
        """Update aggregated location statistics.
        
        Uses 0.01 degree buckets (approximately 1km resolution)
        """
        # Round to ~1km buckets
        lat_bucket = round(latitude, 2)
        lng_bucket = round(longitude, 2)
        
        is_slow = quality_level <= 2  # Slow or very slow
        
        cursor.execute("""
            INSERT INTO location_stats (lat_bucket, lng_bucket, network_type, 
                                        avg_quality, avg_speed_kbps, sample_count, slow_sample_count)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(lat_bucket, lng_bucket, network_type) DO UPDATE SET
                avg_quality = (avg_quality * sample_count + ?) / (sample_count + 1),
                avg_speed_kbps = CASE 
                    WHEN ? IS NOT NULL THEN 
                        CASE WHEN avg_speed_kbps IS NULL THEN ?
                        ELSE (avg_speed_kbps * sample_count + ?) / (sample_count + 1) 
                        END
                    ELSE avg_speed_kbps 
                END,
                sample_count = sample_count + 1,
                slow_sample_count = slow_sample_count + ?,
                last_updated = CURRENT_TIMESTAMP
        """, (
            lat_bucket, lng_bucket, network_type,
            quality_level, speed_kbps, 1 if is_slow else 0,
            quality_level,
            speed_kbps, speed_kbps, speed_kbps,
            1 if is_slow else 0
        ))
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get connection quality statistics.
        
        Args:
            days: Number of days to include in stats
            
        Returns:
            Dictionary with statistics
        """
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Total samples
            cursor.execute("""
                SELECT COUNT(*) FROM connection_samples 
                WHERE timestamp >= ?
            """, (cutoff_ms,))
            total_samples = cursor.fetchone()[0]
            
            # Samples by network type
            cursor.execute("""
                SELECT network_type, COUNT(*) 
                FROM connection_samples 
                WHERE timestamp >= ?
                GROUP BY network_type
            """, (cutoff_ms,))
            samples_by_type = dict(cursor.fetchall())
            
            # Samples by quality level
            cursor.execute("""
                SELECT quality_level, COUNT(*) 
                FROM connection_samples 
                WHERE timestamp >= ?
                GROUP BY quality_level
            """, (cutoff_ms,))
            samples_by_quality_raw = dict(cursor.fetchall())
            samples_by_quality = {
                self.QUALITY_LEVELS.get(k, str(k)): v 
                for k, v in samples_by_quality_raw.items()
            }
            
            # Average speeds by network type
            cursor.execute("""
                SELECT network_type, AVG(download_speed_kbps)
                FROM connection_samples 
                WHERE timestamp >= ? AND download_speed_kbps IS NOT NULL
                GROUP BY network_type
            """, (cutoff_ms,))
            avg_speeds = {k: round(v, 1) for k, v in cursor.fetchall() if v}
            
            # Slow connection locations (quality <= 2)
            cursor.execute("""
                SELECT lat_bucket, lng_bucket, network_type, 
                       avg_quality, avg_speed_kbps, sample_count, slow_sample_count
                FROM location_stats
                WHERE slow_sample_count > 0
                ORDER BY slow_sample_count DESC
                LIMIT 50
            """)
            slow_locations = []
            for row in cursor.fetchall():
                slow_locations.append({
                    "latitude": row[0],
                    "longitude": row[1],
                    "network_type": row[2],
                    "avg_quality": round(row[3], 2) if row[3] else None,
                    "avg_speed_kbps": round(row[4], 1) if row[4] else None,
                    "total_samples": row[5],
                    "slow_samples": row[6],
                    "slow_percentage": round(row[6] / row[5] * 100, 1) if row[5] > 0 else 0
                })
            
            return {
                "total_samples": total_samples,
                "samples_by_network_type": samples_by_type,
                "samples_by_quality": samples_by_quality,
                "average_speeds_by_type": avg_speeds,
                "slow_connection_locations": slow_locations
            }
    
    def get_slow_connection_heatmap(self, min_samples: int = 5) -> List[Dict[str, Any]]:
        """Get locations with high percentages of slow connections.
        
        Args:
            min_samples: Minimum samples to include a location
            
        Returns:
            List of locations with slow connection data
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT lat_bucket, lng_bucket, 
                       SUM(sample_count) as total,
                       SUM(slow_sample_count) as slow,
                       AVG(avg_quality) as quality,
                       AVG(avg_speed_kbps) as speed
                FROM location_stats
                GROUP BY lat_bucket, lng_bucket
                HAVING total >= ?
                ORDER BY (slow * 1.0 / total) DESC
            """, (min_samples,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "latitude": row[0],
                    "longitude": row[1],
                    "total_samples": row[2],
                    "slow_samples": row[3],
                    "slow_percentage": round(row[3] / row[2] * 100, 1) if row[2] > 0 else 0,
                    "avg_quality": round(row[4], 2) if row[4] else None,
                    "avg_speed_kbps": round(row[5], 1) if row[5] else None
                })
            
            return results
    
    def export_for_analysis(self, output_path: Optional[str] = None, days: int = 30) -> str:
        """Export connection data for external analysis.
        
        Args:
            output_path: Output file path (default: connection_data/export.json)
            days: Number of days to export
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = str(self.data_dir / f"connection_export_{datetime.now().strftime('%Y%m%d')}.json")
        
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT device_id, user_id, timestamp, network_type, 
                       quality_level, download_speed_kbps, latitude, longitude
                FROM connection_samples 
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (cutoff_ms,))
            
            samples = []
            for row in cursor.fetchall():
                samples.append({
                    "device_id": row[0],
                    "user_id": row[1],
                    "timestamp": row[2],
                    "timestamp_readable": datetime.fromtimestamp(row[2] / 1000).isoformat(),
                    "network_type": row[3],
                    "quality_level": row[4],
                    "quality_name": self.QUALITY_LEVELS.get(row[4], "unknown"),
                    "download_speed_kbps": row[5],
                    "latitude": row[6],
                    "longitude": row[7]
                })
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "days_included": days,
            "samples": samples
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(samples)} connection samples to {output_path}")
        return output_path


# Global instance
_connection_tracker: Optional[ConnectionQualityTracker] = None


def get_connection_tracker() -> ConnectionQualityTracker:
    """Get or create the global connection tracker instance."""
    global _connection_tracker
    if _connection_tracker is None:
        _connection_tracker = ConnectionQualityTracker()
    return _connection_tracker
