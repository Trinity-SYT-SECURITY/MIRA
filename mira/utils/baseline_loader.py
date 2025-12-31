"""
Baseline prompt loader from datasets.

Loads instruction-following prompts from Hugging Face datasets
(e.g., Alpaca, FLAN) to use as baseline for attack comparison.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import random

try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None


class BaselineLoader:
    """
    Load baseline prompts from instruction-following datasets.
    
    Baseline prompts are normal, safe instructions that represent
    the model's behavior under benign conditions. These are used
    to establish a reference distribution for internal features.
    """
    
    # Supported datasets and their configurations
    DATASET_CONFIGS = {
        "alpaca": {
            "name": "tatsu-lab/alpaca",
            "split": "train",
            "instruction_field": "instruction",
            "description": "Alpaca instruction-following dataset",
        },
        "flan": {
            "name": "google/flan_v2",
            "split": "train",
            "instruction_field": "inputs",  # FLAN uses 'inputs' field
            "description": "FLAN instruction mix dataset",
        },
    }
    
    def __init__(
        self,
        dataset_name: str = "alpaca",
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        """
        Initialize baseline loader.
        
        Args:
            dataset_name: Name of dataset to use ("alpaca" or "flan")
            cache_dir: Cache directory for datasets (default: ~/.cache/huggingface)
            local_dir: Local directory if dataset is already downloaded
                       (e.g., "project/data/raw/alpaca")
        """
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )
        
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(self.DATASET_CONFIGS.keys())}"
            )
        
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self._dataset = None
    
    def load_dataset(self, num_samples: Optional[int] = None, seed: int = 42) -> Dataset:
        """
        Load the dataset.
        
        Args:
            num_samples: Number of samples to load (None for all)
            seed: Random seed for shuffling
            
        Returns:
            Dataset object
        """
        if self._dataset is not None:
            return self._dataset
        
        try:
            # Try loading from local directory first
            if self.local_dir:
                local_path = Path(self.local_dir)
                if not local_path.exists():
                    # Try relative to current working directory
                    local_path = Path.cwd() / self.local_dir
                
                if local_path.exists():
                    # Load from local parquet files (search recursively)
                    parquet_files = list(local_path.rglob("*.parquet"))
                    if not parquet_files:
                        # Also try data/ subdirectory
                        data_dir = local_path / "data"
                        if data_dir.exists():
                            parquet_files = list(data_dir.glob("*.parquet"))
                    
                    if parquet_files:
                        from datasets import load_dataset as load_ds
                        self._dataset = load_ds(
                            "parquet",
                            data_files=[str(f) for f in parquet_files],
                            cache_dir=self.cache_dir,
                        )
                        if isinstance(self._dataset, dict):
                            # If multiple splits, use train
                            self._dataset = self._dataset.get("train", list(self._dataset.values())[0])
                        print(f"    ✓ Loaded baseline dataset from local: {local_path} ({len(parquet_files)} parquet file(s))")
                    else:
                        raise FileNotFoundError(f"No parquet files found in {local_path}")
                else:
                    raise FileNotFoundError(f"Local directory does not exist: {self.local_dir}")
            
            # If local loading failed or no local_dir specified, try Hugging Face
            if self._dataset is None:
                self._dataset = load_dataset(
                    self.config["name"],
                    split=self.config["split"],
                    cache_dir=self.cache_dir,
                )
                print(f"    ✓ Loaded baseline dataset from Hugging Face: {self.config['name']}")
        except Exception as e:
            print(f"    ⚠ Failed to load dataset: {e}")
            print(f"    → Falling back to built-in safe prompts")
            # Fallback to empty dataset - will use built-in prompts
            self._dataset = None
        
        if self._dataset is not None and num_samples:
            # Shuffle and select subset
            self._dataset = self._dataset.shuffle(seed=seed).select(range(num_samples))
        
        return self._dataset
    
    def get_baseline_prompts(
        self,
        num_prompts: int = 50,
        seed: int = 42,
        filter_length: bool = True,
        min_length: int = 10,
        max_length: int = 500,
    ) -> List[str]:
        """
        Extract baseline prompts from the dataset.
        
        Args:
            num_prompts: Number of prompts to return
            seed: Random seed for selection
            filter_length: Whether to filter by length
            min_length: Minimum prompt length (characters)
            max_length: Maximum prompt length (characters)
            
        Returns:
            List of baseline prompt strings
        """
        dataset = self.load_dataset(num_samples=num_prompts * 2, seed=seed)
        
        prompts = []
        
        if dataset is None:
            # Fallback to built-in safe prompts
            from mira.utils.data import load_safe_prompts
            prompts = load_safe_prompts(limit=num_prompts)
            print(f"    ✓ Using {len(prompts)} built-in safe prompts as baseline")
            return prompts
        
        instruction_field = self.config["instruction_field"]
        
        for example in dataset:
            prompt = None
            
            # Try to extract instruction field (Alpaca uses "instruction")
            if instruction_field in example:
                prompt = str(example[instruction_field]).strip()
            elif "instruction" in example:
                prompt = str(example["instruction"]).strip()
            # Alpaca dataset may also have "input" field that can be combined with instruction
            elif "input" in example and example.get("input"):
                # Combine instruction and input if both exist
                inst = str(example.get("instruction", "")).strip()
                inp = str(example["input"]).strip()
                if inst:
                    prompt = f"{inst} {inp}".strip() if inp else inst
                else:
                    prompt = inp
            elif "text" in example:
                prompt = str(example["text"]).strip()
            else:
                # Try first string field that's not empty
                for key, value in example.items():
                    if isinstance(value, str) and len(value.strip()) > min_length:
                        prompt = value.strip()
                        break
            
            if not prompt:
                continue
            
            # Filter by length if requested
            if filter_length:
                if len(prompt) < min_length or len(prompt) > max_length:
                    continue
            
            # Skip empty or very short prompts
            if not prompt or len(prompt.split()) < 3:
                continue
            
            prompts.append(prompt)
            
            if len(prompts) >= num_prompts:
                break
        
        if len(prompts) < num_prompts:
            print(f"    ⚠ Only found {len(prompts)}/{num_prompts} baseline prompts")
            # Fill with built-in safe prompts if needed
            if len(prompts) < num_prompts:
                from mira.utils.data import load_safe_prompts
                additional = load_safe_prompts(limit=num_prompts - len(prompts))
                prompts.extend(additional)
        
        print(f"    ✓ Loaded {len(prompts)} baseline prompts from {self.dataset_name}")
        return prompts
    
    def get_baseline_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        dataset = self.load_dataset()
        
        if dataset is None:
            return {
                "dataset_name": "built-in",
                "num_examples": 0,
                "source": "fallback",
            }
        
        return {
            "dataset_name": self.config["name"],
            "num_examples": len(dataset),
            "source": "huggingface" if not self.local_dir else "local",
            "local_dir": self.local_dir,
        }


def load_baseline_prompts(
    dataset_name: str = "alpaca",
    num_prompts: int = 50,
    local_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """
    Convenience function to load baseline prompts.
    
    Args:
        dataset_name: Dataset to use ("alpaca" or "flan")
        num_prompts: Number of prompts to return
        local_dir: Local directory if dataset is already downloaded
        seed: Random seed
        
    Returns:
        List of baseline prompt strings
    """
    if not HAS_DATASETS:
        # Fallback to built-in safe prompts if datasets library is not available
        from mira.utils.data import load_safe_prompts
        prompts = load_safe_prompts(limit=num_prompts)
        print(f"    ⚠ datasets library not available, using {len(prompts)} built-in safe prompts")
        return prompts
    
    try:
        loader = BaselineLoader(dataset_name=dataset_name, local_dir=local_dir)
        return loader.get_baseline_prompts(num_prompts=num_prompts, seed=seed)
    except Exception as e:
        # Fallback to built-in safe prompts on any error
        from mira.utils.data import load_safe_prompts
        prompts = load_safe_prompts(limit=num_prompts)
        print(f"    ⚠ Failed to load from dataset ({e}), using {len(prompts)} built-in safe prompts")
        return prompts

