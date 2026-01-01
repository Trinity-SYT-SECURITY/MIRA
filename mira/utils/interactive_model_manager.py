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
            project_models_dir: Path to project models directory (default: project/models or from config)
        """
        # Get config file path
        mira_root = Path(__file__).parent.parent.parent
        self.config_file = mira_root / ".mira_config.json"
        
        # Validate and fix config if needed
        self._validate_and_fix_config()
        
        # Load config to get saved models directory
        config = self._load_config()
        
        if project_models_dir is None:
            # Try to use saved directory from config, otherwise use default
            if "models_directory" in config:
                saved_path_str = config["models_directory"]
                saved_platform = config.get("models_directory_platform", "")
                current_platform = self._get_platform_id()
                
                # Check if path was saved on different platform
                # Silently handle cross-platform paths (all paths are dynamically computed)
                
                try:
                    saved_path = Path(saved_path_str)
                    
                    # Try relative path first (cross-platform compatible)
                    if not saved_path.is_absolute():
                        # Relative path - resolve relative to project root
                        project_models_dir = (mira_root / saved_path).resolve()
                    else:
                        # Absolute path - try to use it, but validate
                        project_models_dir = saved_path.resolve()
                    
                    # Validate the path
                    if project_models_dir.exists() or project_models_dir.parent.exists():
                        # Path is valid
                        pass
                    else:
                        # Path doesn't exist - might be from different platform
                        # Silently use default (no hardcoded paths)
                        project_models_dir = mira_root / "project" / "models"
                            
                except (OSError, ValueError) as e:
                    # Path is invalid (e.g., wrong OS path format) - silently use default
                    # No hardcoded paths, all paths are dynamically computed
                    project_models_dir = mira_root / "project" / "models"
            else:
                # Default to project/models relative to MIRA root
                project_models_dir = mira_root / "project" / "models"
        
        self.project_models_dir = Path(project_models_dir)
        
        # Try to create directory, but handle errors gracefully
        try:
            self.project_models_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # If we can't create the directory, use a fallback
            print(f"⚠️  Warning: Cannot create directory {self.project_models_dir}")
            print(f"   Error: {e}")
            # Try to use a directory in the user's home directory as fallback
            try:
                fallback_dir = Path.home() / "MIRA_models"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.project_models_dir = fallback_dir
                print(f"   Using fallback directory: {fallback_dir}")
            except Exception as fallback_error:
                # Last resort: use current directory
                self.project_models_dir = Path.cwd() / "models"
                self.project_models_dir.mkdir(parents=True, exist_ok=True)
                print(f"   Using current directory: {self.project_models_dir}")
        
        # Detect HuggingFace cache directory
        self.hf_cache_dir = self._detect_hf_cache()
    
    def get_required_models_from_registry(self) -> List[Dict[str, Any]]:
        """
        Get required models from MODEL_REGISTRY based on roles.
        GPU-aware: automatically recommends SOTA models when GPU is detected.
        
        Returns:
            List of required model dicts with role information
        """
        from mira.utils.model_manager import MODEL_REGISTRY, get_recommended_models
        from mira.utils.gpu_models import detect_gpu, get_gpu_required_models, get_gpu_models_for_tier
        
        # Detect GPU
        gpu_info = detect_gpu()
        
        if gpu_info["available"]:
            # GPU mode - use SOTA models
            print(f"\n🎮 GPU Detected: {gpu_info['device_name']}")
            print(f"   VRAM: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"   Recommended Tier: {gpu_info['recommended_tier']}")
            
            # Get GPU models for this tier
            if gpu_info['recommended_tier'] != 'cpu':
                gpu_models = get_gpu_models_for_tier(gpu_info['recommended_tier'])
                
                # Convert to required format
                required_models = []
                for model in gpu_models:
                    required_models.append({
                        "name": model["hf_name"],
                        "local_name": model["local_name"],
                        "description": f"{model['description']} ({model['size']})",
                        "role": model["role"],
                        "optional": model.get("tier") != "required",
                        "replaceable_by": model.get("replaceable_by", []),
                        "tier": model.get("tier", "required"),
                        "research_use": model.get("research_use", ""),
                    })
                
                return required_models
        
        # CPU mode - use original logic
        print("\n💻 CPU Mode - Using lightweight models")
        
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
                    "replaceable_by": model.get("replaceable_by", []),
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
                    "replaceable_by": model.get("replaceable_by", []),
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
                        "replaceable_by": model.get("replaceable_by", []),
                    })
        
        return required_models
    
    def get_all_available_models_by_role(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all models from MODEL_REGISTRY grouped by role.
        
        Returns:
            Dict mapping role to list of model dicts
        """
        from mira.utils.model_manager import get_recommended_models
        
        models_by_role = {
            "target": get_recommended_models(role="target"),
            "judge": get_recommended_models(role="judge"),
            "attacker": get_recommended_models(role="attacker"),
        }
        
        # Convert to consistent format
        result = {}
        for role, models in models_by_role.items():
            result[role] = []
            for model in models:
                result[role].append({
                    "name": model["hf_name"],
                    "local_name": model["local_name"],
                    "description": f"{model['description']} ({model['size']})",
                    "role": role,
                    "recommended": model.get("recommended", False),
                })
        
        return result
    
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MIRA configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_config(self, config: Dict[str, Any]):
        """Save MIRA configuration."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _save_models_directory_to_config(self, models_dir: Path):
        """
        Save models directory to config file so program loads from here.
        Uses relative path if within project, absolute path otherwise (with platform detection).
        """
        config = self._load_config()
        mira_root = Path(__file__).parent.parent.parent
        
        try:
            # Try to make path relative to project root for portability
            try:
                resolved_path = models_dir.resolve()
                relative_path = None
                
                # Check if path is within project directory
                try:
                    relative_path = resolved_path.relative_to(mira_root.resolve())
                    # Save as relative path for cross-platform compatibility
                    config["models_directory"] = str(relative_path)
                    config["models_directory_absolute"] = str(resolved_path)  # Keep absolute as backup
                    config["models_directory_platform"] = self._get_platform_id()  # Track which platform saved it
                    self._save_config(config)
                    print(f"✅ Saved models directory to config (relative): {relative_path}")
                    return
                except ValueError:
                    # Path is outside project, save as absolute but with platform info
                    pass
                
                # Path is outside project, save absolute with platform detection
                config["models_directory"] = str(resolved_path)
                config["models_directory_absolute"] = str(resolved_path)
                config["models_directory_platform"] = self._get_platform_id()
                self._save_config(config)
                print(f"✅ Saved models directory to config: {resolved_path}")
                
            except (OSError, ValueError) as e:
                # Can't resolve, save as-is
                config["models_directory"] = str(models_dir)
                config["models_directory_platform"] = self._get_platform_id()
                self._save_config(config)
                print(f"⚠️  Saved models directory (unresolved): {models_dir}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save path to config: {e}")
    
    def _get_platform_id(self) -> str:
        """Get platform identifier for cross-platform path detection."""
        import platform
        import sys
        return f"{platform.system()}_{platform.machine()}_{sys.platform}"
    
    def _validate_and_fix_config(self) -> bool:
        """
        Validate and fix configuration file if needed.
        Handles cross-platform path issues silently - no hardcoded paths.
        
        Returns:
            True if config was fixed, False if no fix needed
        """
        config = self._load_config()
        fixed = False
        mira_root = Path(__file__).parent.parent.parent
        current_platform = self._get_platform_id()
        
        if "models_directory" in config:
            saved_path_str = config["models_directory"]
            saved_platform = config.get("models_directory_platform", "")
            
            # Check if path is relative (cross-platform compatible)
            saved_path = Path(saved_path_str)
            is_relative = not saved_path.is_absolute()
            
            # If saved on different platform
            if saved_platform and saved_platform != current_platform:
                if is_relative:
                    # Relative path - try to validate it
                    try:
                        test_path = (mira_root / saved_path_str).resolve()
                        if test_path.exists() or test_path.parent.exists():
                            # Relative path is valid, keep it (cross-platform compatible)
                            return False
                    except:
                        pass
                
                # Absolute path from different platform - silently convert to relative if possible
                # or remove if not convertible
                try:
                    # Try to see if we can extract a relative path from the absolute path
                    # by comparing with common patterns (e.g., project/models at the end)
                    if "project" in saved_path_str.lower() or "models" in saved_path_str.lower():
                        # Try to use default relative path
                        default_relative = "project/models"
                        test_path = (mira_root / default_relative).resolve()
                        if test_path.parent.exists():
                            # Update to relative path
                            config["models_directory"] = default_relative
                            config["models_directory_platform"] = current_platform
                            if "models_directory_absolute" in config:
                                del config["models_directory_absolute"]
                            self._save_config(config)
                            fixed = True
                            return fixed
                except:
                    pass
                
                # Can't convert, silently remove (will use default path)
                if "models_directory_absolute" in config:
                    del config["models_directory_absolute"]
                if "models_directory_platform" in config:
                    del config["models_directory_platform"]
                del config["models_directory"]
                self._save_config(config)
                fixed = True
            else:
                # Same platform or no platform info, validate path
                try:
                    # Try relative path first (preferred for cross-platform)
                    if is_relative:
                        test_path = (mira_root / saved_path_str).resolve()
                    else:
                        test_path = saved_path.resolve()
                    
                    if not (test_path.exists() or test_path.parent.exists()):
                        # Invalid path - try to convert absolute to relative if within project
                        if not is_relative:
                            try:
                                # Try to make it relative
                                relative = test_path.relative_to(mira_root.resolve())
                                config["models_directory"] = str(relative)
                                config["models_directory_platform"] = current_platform
                                self._save_config(config)
                                fixed = True
                                return fixed
                            except ValueError:
                                # Not within project, remove it
                                pass
                        
                        # Can't fix, silently remove (will use default)
                        if "models_directory_absolute" in config:
                            del config["models_directory_absolute"]
                        if "models_directory_platform" in config:
                            del config["models_directory_platform"]
                        del config["models_directory"]
                        self._save_config(config)
                        fixed = True
                except (OSError, ValueError):
                    # Invalid path format - silently remove (will use default)
                    if "models_directory_absolute" in config:
                        del config["models_directory_absolute"]
                    if "models_directory_platform" in config:
                        del config["models_directory_platform"]
                    del config["models_directory"]
                    self._save_config(config)
                    fixed = True
        
        return fixed
    
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
            dir_name = m["name"]  # Directory name like "HuggingFaceTB--SmolLM2-1.7B-Instruct"
            # Add directory name as-is
            available_names.add(dir_name)
            # Convert directory name to model name format (-- to /)
            model_name_from_dir = dir_name.replace("--", "/")
            available_names.add(model_name_from_dir)
            # Also add reverse conversion (in case it's already in model format)
            available_names.add(dir_name.replace("/", "--"))
        
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
            
            # Check if model is available directly
            if any(variant in available_names for variant in name_variants):
                available.append(req_model)
            else:
                # Check if a replacement model is available
                replaceable_by = req_model.get("replaceable_by", [])
                replacement_found = False
                for replacement in replaceable_by:
                    replacement_variants = [
                        replacement,
                        replacement.replace("/", "--"),
                        replacement.split("/")[-1],
                    ]
                    # Also check local_name of replacement from registry
                    from mira.utils.model_manager import get_model_info
                    repl_info = get_model_info(replacement)
                    if repl_info:
                        replacement_variants.append(repl_info.get("local_name", ""))
                    
                    if any(rv in available_names for rv in replacement_variants):
                        replacement_found = True
                        available.append(req_model)  # Consider as available
                        break
                
                if not replacement_found:
                    missing.append(req_model)
        
        return missing, available
    
    def download_model(self, model_name: str, target_dir: Optional[Path] = None) -> bool:
        """
        Download a model from HuggingFace directly to target directory.
        
        Uses snapshot_download to download directly to target location,
        avoiding HuggingFace cache directory.
        
        Args:
            model_name: Model name in HF format
            target_dir: Target directory (default: project_models_dir)
            
        Returns:
            True if download successful
        """
        try:
            from huggingface_hub import snapshot_download
            
            target_dir = target_dir or self.project_models_dir
            local_name = model_name.replace("/", "--")
            save_path = target_dir / local_name
            
            print(f"📥 Downloading {model_name}...")
            print(f"   Target: {save_path}")
            
            # Ensure target directory exists
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Download directly to target directory using snapshot_download
            # This bypasses the HuggingFace cache and downloads directly to save_path
            snapshot_download(
                repo_id=model_name,
                local_dir=str(save_path),
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
                resume_download=True,  # Resume if download was interrupted
            )
            
            # Verify download by checking for essential files
            if not (save_path / "config.json").exists():
                # Clean up incomplete download
                if save_path.exists():
                    try:
                        shutil.rmtree(save_path)
                        print(f"   🧹 Cleaned up incomplete download")
                    except:
                        pass
                raise FileNotFoundError(f"Download incomplete: config.json not found in {save_path}")
            
            print(f"✅ Downloaded: {model_name}")
            return True
            
        except ImportError:
            # Fallback to old method if huggingface_hub not available
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print(f"⚠️  Using fallback download method (may use cache)...")
                
                target_dir = target_dir or self.project_models_dir
                local_name = model_name.replace("/", "--")
                save_path = target_dir / local_name
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Download with local_dir to force direct download
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_dir=str(save_path),
                    local_files_only=False,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_dir=str(save_path),
                    local_files_only=False,
                )
                
                # Save to ensure files are in target directory
                tokenizer.save_pretrained(str(save_path))
                model.save_pretrained(str(save_path))
                
                print(f"✅ Downloaded: {model_name}")
                return True
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return False
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "Can't load the model" in error_msg or "local directory" in error_msg.lower():
                print(f"❌ Download failed: Model download was interrupted or incomplete")
                print(f"   Error: {error_msg}")
                print(f"   Tip: Try downloading again, or check if the directory {save_path} exists and is accessible")
            else:
                print(f"❌ Download failed: {error_msg}")
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
    
    def migrate_all_hf_models(self, target_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Migrate all models from HuggingFace cache to target directory.
        
        Args:
            target_dir: Target directory (default: project_models_dir)
            
        Returns:
            List of migrated model info dicts
        """
        target_dir = target_dir or self.project_models_dir
        hf_models = self.scan_hf_cache()
        migrated = []
        
        print(f"\n📦 Migrating {len(hf_models)} models from HuggingFace cache...")
        print(f"   From: {self.hf_cache_dir}")
        print(f"   To:   {target_dir}\n")
        
        for model in hf_models:
            if self.migrate_model(model, target_dir=target_dir):
                migrated.append(model)
        
        return migrated
    
    def interactive_model_setup(self) -> Dict[str, Any]:
        """
        Interactive model setup at startup.
        Asks user where to load models from, then checks for required models.
        
        Returns:
            Dict with setup results
        """
        print("\n" + "=" * 70)
        print("🔧 Model Management System")
        print("=" * 70)
        
        # Scan HuggingFace cache first to check for models
        hf_models = self.scan_hf_cache()
        
        # Step 1: Ask user where to load models from
        print("\n📂 Where should MIRA load models from?")
        print(f"  1. Current directory: {self.project_models_dir}")
        print(f"  2. Custom directory")
        if hf_models:
            print(f"  3. Migrate models from HuggingFace cache ({len(hf_models)} models found)")
        print(f"  4. Use default: project/models")
        
        try:
            load_choice = input("\nYour choice (1-4, default: 1): ").strip()
            if not load_choice:
                load_choice = "1"
        except:
            load_choice = "1"
        
        if load_choice == "2":
            custom_dir = input("Enter directory path: ").strip()
            if custom_dir:
                self.project_models_dir = Path(custom_dir).expanduser().resolve()
                self.project_models_dir.mkdir(parents=True, exist_ok=True)
                self._save_models_directory_to_config(self.project_models_dir)
                print(f"✅ Models directory set to: {self.project_models_dir}")
                
                # If there are HF cache models, ask if user wants to migrate
                if hf_models:
                    migrate_choice = input(f"\nMigrate {len(hf_models)} models from HuggingFace cache to this directory? (y/n, default: y): ").strip().lower()
                    if not migrate_choice or migrate_choice == 'y':
                        self.migrate_all_hf_models(self.project_models_dir)
        elif load_choice == "3" and hf_models:
            # User wants to migrate from HF cache
            print(f"\n💾 Found {len(hf_models)} models in HuggingFace cache: {self.hf_cache_dir}")
            print("\nWhere should we migrate these models to?")
            print(f"  1. Current directory: {self.project_models_dir}")
            print(f"  2. Custom directory")
            
            migrate_choice = input("Choice (1-2, default: 1): ").strip()
            if not migrate_choice:
                migrate_choice = "1"
            
            if migrate_choice == "1":
                self.migrate_all_hf_models(self.project_models_dir)
            elif migrate_choice == "2":
                custom_target = input("Enter target directory path: ").strip()
                target_dir = Path(custom_target).expanduser().resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                self.migrate_all_hf_models(target_dir)
                self.project_models_dir = target_dir
                self._save_models_directory_to_config(target_dir)
        elif load_choice == "4":
            # Use default
            mira_root = Path(__file__).parent.parent.parent
            self.project_models_dir = mira_root / "project" / "models"
            self.project_models_dir.mkdir(parents=True, exist_ok=True)
            
            # If there are HF cache models, ask if user wants to migrate
            if hf_models:
                migrate_choice = input(f"\nMigrate {len(hf_models)} models from HuggingFace cache to default directory? (y/n, default: y): ").strip().lower()
                if not migrate_choice or migrate_choice == 'y':
                    self.migrate_all_hf_models(self.project_models_dir)
        
        # Scan models in the selected directory
        project_models = self.scan_project_models()
        
        # If no models found in selected directory, check project root's models folder
        if not project_models:
            mira_root = Path(__file__).parent.parent.parent
            root_models_dir = mira_root / "models"
            if root_models_dir.exists() and root_models_dir != self.project_models_dir:
                # Temporarily scan root models directory
                original_dir = self.project_models_dir
                self.project_models_dir = root_models_dir
                root_models = self.scan_project_models()
                self.project_models_dir = original_dir
                
                if root_models:
                    print(f"\n💡 Found {len(root_models)} models in: {root_models_dir}")
                    print(f"   But you selected: {self.project_models_dir}")
                    use_root = input(f"   Use models from {root_models_dir} instead? (y/n, default: y): ").strip().lower()
                    if not use_root or use_root == 'y':
                        self.project_models_dir = root_models_dir
                        self._save_models_directory_to_config(root_models_dir)
                        project_models = root_models
        
        print(f"\n📁 Models directory: {self.project_models_dir}")
        print(f"   Models found: {len(project_models)}")
        
        if project_models:
            print("\n   Available models:")
            for model in project_models[:10]:  # Show first 10
                print(f"      • {model['name']} ({model['size_mb']:.1f} MB)")
            if len(project_models) > 10:
                print(f"      ... and {len(project_models) - 10} more")
        
        # Step 2: Check for required models and let user select which to download
        missing_required, available_required = self.check_required_models()
        
        if missing_required:
            print("\n⚠️  Missing Models:")
            
            # Group by role for better display
            by_role = {}
            for req in missing_required:
                role = req.get("role", "other")
                if role not in by_role:
                    by_role[role] = []
                by_role[role].append(req)
            
            # Display by role and let user select
            role_names = {
                "target": "Target Models (Victim models for testing)",
                "judge": "Judge Models (Evaluation & safety)",
                "attacker": "Attacker Models (Attack generation)"
            }
            
            selected_models = []
            
            for role, models in by_role.items():
                print(f"\n   {role_names.get(role, role.upper())}:")
                for i, req in enumerate(models, 1):
                    optional_tag = " [optional]" if req.get("optional") else " [required]"
                    print(f"      [{i}] {req['name']}{optional_tag}")
                    print(f"          {req['description']}")
                
                # Let user select models for this role
                if role == "judge":
                    # Judge models are usually all required
                    print(f"\n   Select judge models to download (comma-separated, e.g., 1,2 or 'all' for all):")
                    selection = input("   Your selection (default: all): ").strip().lower()
                    if not selection or selection == "all":
                        selected_models.extend(models)
                    else:
                        try:
                            indices = [int(x.strip()) - 1 for x in selection.split(",")]
                            for idx in indices:
                                if 0 <= idx < len(models):
                                    selected_models.append(models[idx])
                        except ValueError:
                            print(f"   ⚠️  Invalid input, skipping {role} models")
                else:
                    # For target and attacker, let user choose
                    print(f"\n   Select {role} models to download (comma-separated, e.g., 1,2,3 or 'all' for all, 'skip' to skip):")
                    selection = input("   Your selection (default: all): ").strip().lower()
                    if selection == "skip":
                        continue
                    elif not selection or selection == "all":
                        selected_models.extend(models)
                    else:
                        try:
                            indices = [int(x.strip()) - 1 for x in selection.split(",")]
                            for idx in indices:
                                if 0 <= idx < len(models):
                                    selected_models.append(models[idx])
                        except ValueError:
                            print(f"   ⚠️  Invalid input, skipping {role} models")
            
            if selected_models:
                print(f"\n📋 Selected {len(selected_models)} models to download:")
                for model in selected_models:
                    print(f"   • {model['name']} ({model.get('role', 'unknown')})")
                
                download_choice = input("\n\nDownload selected models? (y/n, default: y): ").strip().lower()
                if not download_choice or download_choice == 'y':
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
                        
                        # Save custom directory to config so program loads from here
                        self._save_models_directory_to_config(download_dir)
                        # Update project_models_dir for this session
                        self.project_models_dir = download_dir
                    
                    if download_dir:
                        print(f"\n📥 Downloading to: {download_dir}\n")
                        downloaded_count = 0
                        for model in selected_models:
                            if self.download_model(model["name"], download_dir):
                                downloaded_count += 1
                            print()  # Blank line between downloads
                        
                        print(f"\n✅ Downloaded {downloaded_count} models")
        else:
            print("\n✅ All required models are available")
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Model Setup Complete")
        print("=" * 70)
        print(f"   Models directory: {self.project_models_dir}")
        print(f"   Models found: {len(project_models)}")
        print(f"   Required models: {len(available_required)}/{len(available_required) + len(missing_required)}")
        print("=" * 70)
        
        print("\nPress Enter to continue...")
        input()
        
        return {
            "project_models": project_models,
            "hf_models": hf_models,
            "migrated": [],
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
