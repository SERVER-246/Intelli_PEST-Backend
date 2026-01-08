"""
Migration Script: Unflag Users and Redistribute Images
======================================================
This script:
1. Unflagged all flagged users (since they are trusted experts)
2. Redistributes images from flagged/ folders to appropriate categories
3. Migrates JSON user data to SQLite database
4. Resets trust scores to 100 for all users

Run this once after the flagging system changes.
"""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime

# Add parent to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def unflag_all_users(users_dir: Path) -> int:
    """Unflag all users in JSON files."""
    count = 0
    for filepath in users_dir.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            if data.get("is_flagged", False):
                original_reason = data.get("flag_reason", "Unknown")
                data["is_flagged"] = False
                data["flag_reason"] = f"Unflagged (was: {original_reason}) - Users are trusted experts"
                data["trust_score"] = 100.0  # Reset to full trust
                
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                
                print(f"  Unflagged: {data['user_id']} (was: {original_reason})")
                count += 1
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
    
    return count


def redistribute_flagged_images(images_dir: Path, metadata_dir: Path) -> dict:
    """
    Move images from flagged/ to appropriate folders based on user corrections.
    """
    flagged_dir = images_dir / "flagged"
    if not flagged_dir.exists():
        return {"moved": 0, "skipped": 0}
    
    stats = {"moved": 0, "skipped": 0, "by_class": {}}
    
    # Process each user's flagged folder
    for user_dir in flagged_dir.iterdir():
        if not user_dir.is_dir():
            continue
        
        print(f"  Processing flagged images from user: {user_dir.name}")
        
        for image_file in user_dir.glob("*.jpg"):
            image_hash = image_file.stem.split("_")[0]  # Get hash from filename
            
            # Find metadata for this image
            metadata_file = metadata_dir / f"{image_hash}.json"
            
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Determine where to move based on feedback
                if metadata.get("feedback_status") == "correct":
                    # Move to correct/{predicted_class}
                    target_class = metadata.get("predicted_class", "unknown")
                    target_dir = images_dir / "correct" / target_class
                elif metadata.get("feedback_status") == "corrected":
                    # Move to corrected/{corrected_class}
                    target_class = metadata.get("corrected_class", "unknown")
                    target_dir = images_dir / "corrected" / target_class
                else:
                    # Unverified - move to corrected based on any correction info
                    if metadata.get("corrected_class"):
                        target_class = metadata["corrected_class"]
                        target_dir = images_dir / "corrected" / target_class
                    else:
                        # No correction info - keep with predicted class as unverified
                        target_class = metadata.get("predicted_class", "unknown")
                        target_dir = images_dir / "unverified" / datetime.now().strftime("%Y-%m-%d")
                
                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / image_file.name
                
                try:
                    shutil.move(str(image_file), str(target_path))
                    
                    # Update metadata with new path
                    metadata["image_path"] = str(target_path)
                    metadata["is_trusted_submission"] = True  # Mark as trusted now
                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    stats["moved"] += 1
                    stats["by_class"][target_class] = stats["by_class"].get(target_class, 0) + 1
                    print(f"    Moved {image_file.name} -> {target_class}/")
                except Exception as e:
                    print(f"    Error moving {image_file.name}: {e}")
                    stats["skipped"] += 1
            else:
                print(f"    No metadata for {image_file.name}, skipping")
                stats["skipped"] += 1
        
        # Remove empty user directory
        try:
            if not any(user_dir.iterdir()):
                user_dir.rmdir()
        except:
            pass
    
    return stats


def migrate_to_database(base_dir: Path):
    """Initialize database and migrate data."""
    try:
        from inference_server.feedback.database import init_database_manager
        
        db_path = base_dir / "intellipest.db"
        db = init_database_manager(str(db_path))
        
        # Migrate JSON data
        result = db.migrate_from_json(
            str(base_dir / "users"),
            str(base_dir / "metadata")
        )
        
        # Unflag all users in database
        unflagged = db.unflag_all_users("Flagging disabled - all users are trusted experts")
        
        return {"database": str(db_path), "migrated": result, "unflagged_in_db": unflagged}
    except Exception as e:
        print(f"  Database migration error: {e}")
        return None


def main():
    print("=" * 60)
    print("MIGRATION: Disable Flagging & Redistribute Images")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent / "feedback_data"
    users_dir = base_dir / "users"
    images_dir = base_dir / "images"
    metadata_dir = base_dir / "metadata"
    
    print(f"\nBase directory: {base_dir}")
    print(f"Users directory: {users_dir}")
    print(f"Images directory: {images_dir}")
    print()
    
    # Step 1: Unflag all users
    print("STEP 1: Unflagging all users...")
    print("-" * 40)
    if users_dir.exists():
        unflagged_count = unflag_all_users(users_dir)
        print(f"Total unflagged: {unflagged_count} users")
    else:
        print("  No users directory found")
    print()
    
    # Step 2: Redistribute flagged images
    print("STEP 2: Redistributing flagged images...")
    print("-" * 40)
    if images_dir.exists():
        redistrib_stats = redistribute_flagged_images(images_dir, metadata_dir)
        print(f"Moved: {redistrib_stats['moved']} images")
        print(f"Skipped: {redistrib_stats['skipped']} images")
        if redistrib_stats.get("by_class"):
            print("By class:")
            for cls, count in redistrib_stats["by_class"].items():
                print(f"  {cls}: {count}")
    else:
        print("  No images directory found")
    print()
    
    # Step 3: Migrate to database
    print("STEP 3: Migrating to SQLite database...")
    print("-" * 40)
    db_result = migrate_to_database(base_dir)
    if db_result:
        print(f"Database created: {db_result['database']}")
        print(f"Migrated users: {db_result['migrated'].get('users', 0)}")
        print(f"Migrated metadata: {db_result['migrated'].get('metadata', 0)}")
        print(f"Unflagged in DB: {db_result['unflagged_in_db']}")
    print()
    
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print("\nChanges made:")
    print("1. All users have been unflagged")
    print("2. Trust scores reset to 100 for all users")
    print("3. Flagged images redistributed based on user corrections")
    print("4. Data migrated to SQLite database")
    print("\nNote: The FLAGGING_ENABLED flag in user_tracker.py is set to False.")
    print("All users are now treated as trusted experts.")


if __name__ == "__main__":
    main()
