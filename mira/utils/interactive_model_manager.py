"""
Interactive Model Management System.

Detects models in HuggingFace cache and project/models directory,
offers to migrate models for unified management, and handles downloads.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json


class ModelManager:
    """Manages model storage, detection, and migration."""
    
    def __init__(self, project_models_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            project_models_dir: Path to project models directory (default: project/models)
        """
        if project_models_dir is None:
            # Default to project/models relative to MIRA root
            mira_root = Path(__file__).parent.parent.parent
            project_models_dir = mira_root / "project" / "models"
        
        self.project_models_dir = Path(project_models_dir)
        self.project_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect HuggingFace cache directory
        self.hf_cache_dir = self._detect_hf_cache()
    
    def get_required_models_from_registry(self) -> List[Dict[str, Any]]:
        """
        Get required models from MODEL_REGISTRY based on roles.
        
        Returns:
            List of required model dicts with role information
        """
        from mira.utils.model_manager import MODEL_REGISTRY, get_recommended_models
        
        required_models = []
        
        # Get recommended models for each role
        target_models = get_recommended_models(role="target")
        judge_models = get_recommended_models(role="judge")
        attacker_models = get_recommended_models(role="attacker")
        
        # Add recommended target models (at least one required)
        if target_models:
            for model in target_models[:3]:  # Top 3 recommended
                required_models.append({
                    "name": model["hf_name"],
                    "local_name": model["local_name"],
                    "description": f"{model['description']} ({model['size']})",
                    "role": "target",
                    "optional": not model.get("recommended", False),
                })
        
        # Add all recommended judge models (critical for evaluation)
        for model in judge_models:
            if model.get("recommended"):
                required_models.append({
                    "name": model["hf_name"],
                    "local_name": model["local_name"],
                    "description": f"{model['description']} ({model['size']})",
                    "role": "judge",
                    "optional": False,  # Judges are always required
                })
        
        # Add recommended attacker models (optional)
        if attacker_models:
            for model in attacker_models[:1]:  # At least one attacker
                if model.get("recommended"):
                    required_models.append({
                        "name": model["hf_name"],
                        "local_name": model["local_name"],
                        "description": f"{model['description']} ({model['size']})",
                        "role": "attacker",
                        "optional": True,
                    })
        
        return required_models
    
    def _detect_hf_cache(self) -> Optional[Path]:
        """Detect HuggingFace cache directory."""
        # Try common locations
        possible_locations = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
            Path(os.environ.get("TRANSFORMERS_CACHE", "")) if os.environ.get("TRANSFORMERS_CACHE") else None,
        ]
        
        for loc in possible_locations:
            if loc and loc.exists():
                return loc
        
        return None
    
    def scan_project_models(self) -> List[Dict[str, Any]]:
        """
        Scan project/models directory for existing models.
        
        Returns:
            List of model info dicts
        """
        models = []
        
        if not self.project_models_dir.exists():
            return models
        
        for item in self.project_models_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid model directory
                has_config = (item / "config.json").exists()
                has_model = any([
                    (item / "pytorch_model.bin").exists(),
                    (item / "model.safetensors").exists(),
                    (item / "pytorch_model.bin.index.json").exists(),
                ])
                
                if has_config or has_model:
                    size = self._get_dir_size(item)
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "size_mb": size / (1024 * 1024),
                        "location": "project",
                    })
        
        return models
    
    def scan_hf_cache(self) -> List[Dict[str, Any]]:
        """
        Scan HuggingFace cache for downloaded models.
        
        Returns:
            List of model info dicts
        """
        models = []
        
        if not self.hf_cache_dir or not self.hf_cache_dir.exists():
            return models
        
        # HF cache structure: models--org--name/snapshots/hash/
        for item in self.hf_cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                # Extract model name
                model_name = item.name.replace("models--", "").replace("--", "/")
                
                # Find latest snapshot
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                        size = self._get_dir_size(latest_snapshot)
                        
                        models.append({
                            "name": model_name,
                            "local_name": item.name.replace("models--", ""),
                            "path": str(latest_snapshot),
                            "cache_path": str(item),
                            "size_mb": size / (1024 * 1024),
                            "location": "hf_cache",
                        })
        
        return models
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (PermissionError, OSError):
            pass
        return total
    
    def check_required_models(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Check which required models are missing using MODEL_REGISTRY.
        
        Returns:
            Tuple of (missing_models, available_models)
        """
        # Get required models from registry
        required_models = self.get_required_models_from_registry()
        
        # Get all available models (project + HF cache)
        project_models = self.scan_project_models()
        hf_models = self.scan_hf_cache()
        
        available_names = set()
        for m in project_models:
            available_names.add(m["name"])
            available_names.add(m["name"].replace("/", "--"))
        
        for m in hf_models:
            available_names.add(m["name"])
            available_names.add(m["local_name"])
        
        missing = []
        available = []
        
        for req_model in required_models:
            model_name = req_model["name"]
            local_name = req_model.get("local_name", model_name.replace("/", "--"))
            
            # Check various name formats
            name_variants = [
                model_name,
                model_name.replace("/", "--"),
                local_name,
                model_name.split("/")[-1],  # Just model name without org
            ]
            
            if any(variant in available_names for variant in name_variants):
                available.append(req_model)
            else:
                missing.append(req_model)
        
        return missing, available
    
    def download_model(self, model_name: str, target_dir: Optional[Path] = None) -> bool:
        """
        Download a model from HuggingFace.
        
        Args:
            model_name: Model name in HF format
            target_dir: Target directory (default: project_models_dir)
            
        Returns:
            True if download successful
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            target_dir = target_dir or self.project_models_dir
            local_name = model_name.replace("/", "--")
            save_path = target_dir / local_name
            
            print(f"📥 Downloading {model_name}...")
            print(f"   Target: {save_path}")
            
            # Download model and tokenizer
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save to target directory
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"✅ Downloaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def migrate_model(
        self,
        model_info: Dict[str, Any],
        target_dir: Optional[Path] = None,
        force: bool = False
    ) -> bool:
        """
        Migrate a model from HF cache to target directory.
        
        Args:
            model_info: Model info dict from scan_hf_cache()
            target_dir: Target directory (default: project_models_dir)
            force: Force migration even if model exists in target
            
        Returns:
            True if migration successful
        """
        if model_info["location"] != "hf_cache":
            print(f"❌ Model is not in HF cache, cannot migrate")
            return False
        
        target_dir = target_dir or self.project_models_dir
        source_path = Path(model_info["path"])
        dest_name = model_info["local_name"]
        dest_path = target_dir / dest_name
        
        # Check if already exists
        if dest_path.exists() and not force:
            print(f"⚠️  Model already exists in project: {dest_name}")
            return False
        
        try:
            print(f"📦 Migrating {model_info['name']} ({model_info['size_mb']:.1f} MB)...")
            print(f"   From: {source_path}")
            print(f"   To:   {dest_path}")
            
            # Copy directory
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path, symlinks=True)
            
            print(f"✅ Migration complete: {dest_name}")
            return True
        
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def interactive_model_setup(self) -> Dict[str, Any]:
        """
        Interactive model setup at startup.
        
        Returns:
            Dict with setup results
        """
        print("\n" + "=" * 70)
        print("🔧 Model Management System")
        print("=" * 70)
        
        # Scan both locations
        project_models = self.scan_project_models()
        hf_models = self.scan_hf_cache()
        
        # Show current status
        print(f"\n📁 Project models directory: {self.project_models_dir}")
        print(f"   Models found: {len(project_models)}")
        
        if project_models:
            print("\n   Current models:")
            for model in project_models[:10]:  # Show first 10
                print(f"      • {model['name']} ({model['size_mb']:.1f} MB)")
            if len(project_models) > 10:
                print(f"      ... and {len(project_models) - 10} more")
        
        if self.hf_cache_dir:
            print(f"\n💾 HuggingFace cache: {self.hf_cache_dir}")
            print(f"   Models found: {len(hf_models)}")
        else:
            print(f"\n💾 HuggingFace cache: Not found")
            hf_models = []
        
        # Find models in HF cache but not in project
        project_names = {m["name"] for m in project_models}
        hf_only_models = [m for m in hf_models if m["local_name"] not in {p["name"] for p in project_models}]
        
        if not hf_only_models:
            print("\n✅ All HuggingFace models are already in project directory")
            print("\nPress Enter to continue...")
            input()
            return {
                "project_models": project_models,
                "hf_models": hf_models,
                "migrated": [],
            }
        
        # Check for required models
        missing_required, _ = self.check_required_models()
        
        if missing_required:
            print("\n⚠️  Missing Required Models:")
            
            # Group by role for better display
            by_role = {}
            for req in missing_required:
                role = req.get("role", "other")
                if role not in by_role:
                    by_role[role] = []
                by_role[role].append(req)
            
            # Display by role
            role_names = {
                "target": "Target Models (Victim models for testing)",
                "judge": "Judge Models (Evaluation & safety)",
                "attacker": "Attacker Models (Attack generation)"
            }
            
            for role, models in by_role.items():
                print(f"\n   {role_names.get(role, role.upper())}:")
                for req in models:
                    optional_tag = " [optional]" if req.get("optional") else " [required]"
                    print(f"      • {req['name']}{optional_tag}")
                    print(f"        {req['description']}")
            
            download_choice = input("\n\nDownload missing required models? (y/n): ").strip().lower()
            if download_choice == 'y':
                # Ask for download location
                print("\nWhere should we download the models?")
                print(f"  1. Project directory (recommended): {self.project_models_dir}")
                print(f"  2. Custom directory")
                print(f"  3. Cancel download")
                
                loc_choice = input("Choice (1-3): ").strip()
                
                download_dir = None
                if loc_choice == "1":
                    download_dir = self.project_models_dir
                elif loc_choice == "2":
                    custom_dir = input("Enter directory path: ").strip()
                    download_dir = Path(custom_dir).expanduser().resolve()
                    download_dir.mkdir(parents=True, exist_ok=True)
                
                if download_dir:
                    print(f"\n📥 Downloading to: {download_dir}\n")
                    downloaded_count = 0
                    for req in missing_required:
                        # Only auto-download required models
                        if not req.get("optional"):
                            if self.download_model(req["name"], download_dir):
                                downloaded_count += 1
                            print()  # Blank line between downloads
                    
                    # Ask about optional models
                    optional_models = [m for m in missing_required if m.get("optional")]
                    if optional_models:
                        print(f"\n{len(optional_models)} optional models available:")
                        for opt in optional_models:
                            print(f"   • {opt['name']} ({opt['role']})")
                        
                        opt_choice = input("\nDownload optional models too? (y/n): ").strip().lower()
                        if opt_choice == 'y':
                            for opt in optional_models:
                                if self.download_model(opt["name"], download_dir):
                                    downloaded_count += 1
                                print()
                    
                    print(f"\n✅ Downloaded {downloaded_count} models")
        
        # Offer migration
        print(f"\n🔄 Found {len(hf_only_models)} models in HuggingFace cache not in project:")
        for i, model in enumerate(hf_only_models, 1):
            print(f"   {i}. {model['name']} ({model['size_mb']:.1f} MB)")
        
        print("\n" + "=" * 70)
        print("Options:")
        print("  1. Migrate all models to project directory")
        print("  2. Select specific models to migrate")
        print("  3. Skip migration (use HuggingFace cache)")
        print("  4. Change project models directory")
        print("=" * 70)
        
        choice = input("\nYour choice (1-4): ").strip()
        
        migrated = []
        
        if choice == "1":
            # Migrate all - ask for target directory
            print("\nWhere should we migrate the models to?")
            print(f"  1. Current project directory: {self.project_models_dir}")
            print(f"  2. Custom directory")
            
            target_choice = input("Choice (1-2): ").strip()
            
            target_dir = self.project_models_dir
            if target_choice == "2":
                custom_target = input("Enter target directory path: ").strip()
                target_dir = Path(custom_target).expanduser().resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Target directory set to: {target_dir}")
            
            print(f"\n📦 Migrating {len(hf_only_models)} models to {target_dir}...")
            for model in hf_only_models:
                if self.migrate_model(model, target_dir=target_dir):
                    migrated.append(model)
        
        elif choice == "2":
            # Select specific models
            print("\nEnter model numbers to migrate (comma-separated, e.g., 1,3,5):")
            selection = input("Models: ").strip()
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                for idx in indices:
                    if 0 <= idx < len(hf_only_models):
                        if self.migrate_model(hf_only_models[idx]):
                            migrated.append(hf_only_models[idx])
            except ValueError:
                print("❌ Invalid input, skipping migration")
        
        elif choice == "4":
            # Change directory
            new_dir = input("\nEnter new project models directory path: ").strip()
            new_path = Path(new_dir).expanduser().resolve()
            new_path.mkdir(parents=True, exist_ok=True)
            self.project_models_dir = new_path
            print(f"✅ Project models directory changed to: {new_path}")
            
            # Re-run setup
            return self.interactive_model_setup()
        
        else:
            print("\n⏭️  Skipping migration, using HuggingFace cache")
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Model Setup Complete")
        print("=" * 70)
        print(f"   Project models: {len(project_models) + len(migrated)}")
        print(f"   Migrated: {len(migrated)}")
        print(f"   HuggingFace cache: {len(hf_models)}")
        print("=" * 70)
        
        print("\nPress Enter to continue...")
        input()
        
        return {
            "project_models": project_models + migrated,
            "hf_models": hf_models,
            "migrated": migrated,
        }
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get path to model, checking project directory first, then HF cache.
        
        Args:
            model_name: Model name (HuggingFace format or local name)
            
        Returns:
            Path to model directory, or None if not found
        """
        # Check project directory first
        # Try direct match
        project_path = self.project_models_dir / model_name
        if project_path.exists():
            return str(project_path)
        
        # Try with -- separator (HF format)
        hf_format = model_name.replace("/", "--")
        project_path = self.project_models_dir / hf_format
        if project_path.exists():
            return str(project_path)
        
        # Check HF cache
        if self.hf_cache_dir:
            cache_name = f"models--{hf_format}"
            cache_path = self.hf_cache_dir / cache_name / "snapshots"
            if cache_path.exists():
                snapshots = list(cache_path.iterdir())
                if snapshots:
                    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                    return str(latest)
        
        return None


def run_model_setup() -> Dict[str, Any]:
    """
    Run interactive model setup at startup.
    
    Returns:
        Dict with setup results
    """
    manager = ModelManager()
    return manager.interactive_model_setup()


if __name__ == "__main__":
    # Test the model manager
    run_model_setup()
