#!/usr/bin/env python3
"""
Cleanup script for old result directories.

Usage:
    python scripts/cleanup_results.py --keep 5    # Keep last 5 results
    python scripts/cleanup_results.py --archive  # Move old results to archive
    python scripts/cleanup_results.py --list      # List all results
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime


def list_results(results_dir: Path):
    """List all result directories sorted by date."""
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    results = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            try:
                # Parse timestamp from directory name
                date_str = item.name.replace("run_", "")
                dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                results.append((dt, item))
            except ValueError:
                continue
    
    # Sort by date (newest first)
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def cleanup_old_results(results_dir: Path, keep: int = 5):
    """Delete old result directories, keeping only the most recent ones."""
    results = list_results(results_dir)
    
    if len(results) <= keep:
        print(f"Only {len(results)} results found, keeping all.")
        return
    
    to_delete = results[keep:]
    print(f"Found {len(results)} results. Keeping {keep} most recent.")
    print(f"Deleting {len(to_delete)} old results...")
    
    for dt, path in to_delete:
        print(f"  Deleting: {path.name} ({dt.strftime('%Y-%m-%d %H:%M:%S')})")
        shutil.rmtree(path)
    
    print(f"✓ Cleaned up {len(to_delete)} old result directories.")


def archive_old_results(results_dir: Path, keep: int = 5, archive_dir: Path = None):
    """Move old result directories to archive."""
    if archive_dir is None:
        archive_dir = results_dir.parent / "results_archive"
    
    archive_dir.mkdir(exist_ok=True)
    
    results = list_results(results_dir)
    
    if len(results) <= keep:
        print(f"Only {len(results)} results found, nothing to archive.")
        return
    
    to_archive = results[keep:]
    print(f"Found {len(results)} results. Keeping {keep} most recent.")
    print(f"Archiving {len(to_archive)} old results to {archive_dir}...")
    
    for dt, path in to_archive:
        dest = archive_dir / path.name
        print(f"  Archiving: {path.name} -> {dest}")
        shutil.move(str(path), str(dest))
    
    print(f"✓ Archived {len(to_archive)} old result directories.")


def main():
    parser = argparse.ArgumentParser(description="Clean up old result directories")
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of recent results to keep (default: 5)"
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Move old results to archive instead of deleting"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Archive directory (default: results_archive/)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all result directories"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results/)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        results = list_results(args.results_dir)
        print(f"\nFound {len(results)} result directories:\n")
        for i, (dt, path) in enumerate(results, 1):
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"{i:2d}. {path.name}")
            print(f"    Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Size: {size_mb:.2f} MB")
        return
    
    if args.archive:
        archive_old_results(args.results_dir, args.keep, args.archive_dir)
    else:
        # Confirm before deleting
        results = list_results(args.results_dir)
        if len(results) > args.keep:
            to_delete = results[args.keep:]
            print(f"\n⚠️  WARNING: This will DELETE {len(to_delete)} result directories!")
            print("\nDirectories to be deleted:")
            for dt, path in to_delete:
                print(f"  - {path.name} ({dt.strftime('%Y-%m-%d %H:%M:%S')})")
            
            response = input("\nContinue? (yes/no): ")
            if response.lower() in ['yes', 'y']:
                cleanup_old_results(args.results_dir, args.keep)
            else:
                print("Cancelled.")
        else:
            print("No old results to clean up.")


if __name__ == "__main__":
    main()

