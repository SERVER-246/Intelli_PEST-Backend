"""
Data Counter Utility
Location: src/utils/data_counter.py

Utility for counting images in dataset folders.
"""

import os
from pathlib import Path

def count_images(folder_path):
    """
    Count all image files in a folder and its subfolders.
    
    Args:
        folder_path: Path to the folder to search
    
    Returns:
        Dictionary with total count and breakdown by subfolder
    """
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico'}
    
    total_count = 0
    folder_counts = {}
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        count = 0
        for file in files:
            # Check if file has an image extension (case-insensitive)
            if os.path.splitext(file)[1].lower() in image_extensions:
                count += 1
                total_count += 1
        
        if count > 0:
            folder_counts[root] = count
    
    return total_count, folder_counts

if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter folder path: ")
    
    try:
        total, breakdown = count_images(folder_path)
        
        print(f"Total images found: {total}\n")
        print("Breakdown by folder:")
        print("-" * 50)
        
        for folder, count in breakdown.items():
            print(f"{folder}: {count} images")
            
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found!")
    except PermissionError:
        print(f"Error: Permission denied to access '{folder_path}'!")
