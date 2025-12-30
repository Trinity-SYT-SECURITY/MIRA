"""
Experiment logging and data recording.

Records all experiment data in structured format for
automatic chart generation and research reporting.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ExperimentRecord:
    """Single experiment record for logging."""
    timestamp: str
    experiment_id: str
    model_name: str
    prompt: str
    attack_type: Optional[str]
    suffix: Optional[str]
    response: Optional[str]  # Full response (prompt + suffix + generated)
    generated_text: Optional[str]  # Only the model-generated part
    success: Optional[bool]
    metrics: Dict[str, float]
    layer_data: Optional[Dict[str, Any]]
    attention_data: Optional[Dict[str, Any]]


class ExperimentLogger:
    """
    Logger for recording experiment data.
    
    Saves structured data in multiple formats for
    easy analysis and visualization.
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize experiment logger.
        
        Args:
            output_dir: Directory for output files
            experiment_name: Name for this experiment run
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name with timestamp
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create subdirectories
        self.data_dir = self.output_dir / experiment_name / "data"
        self.charts_dir = self.output_dir / experiment_name / "charts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.records: List[ExperimentRecord] = []
        self.metrics_history: List[Dict[str, Any]] = []
    
    def log_attack(
        self,
        model_name: str,
        prompt: str,
        attack_type: str,
        suffix: str,
        response: str,
        success: bool,
        metrics: Dict[str, float],
        layer_data: Optional[Dict[str, Any]] = None,
        attention_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single attack attempt.
        
        Args:
            model_name: Name of the model
            prompt: Original prompt
            attack_type: Type of attack used
            suffix: Adversarial suffix
            response: Full model response (includes prompt + suffix + generated)
            success: Whether attack succeeded
            metrics: Dictionary of metric values
            layer_data: Optional layer-wise activation data
            attention_data: Optional attention pattern data
        """
        # Extract generated text (remove prompt and suffix from response)
        generated_text = None
        if response:
            # Calculate the expected prefix length
            prefix = prompt
            if suffix:
                prefix = prompt + " " + suffix
            
            # Remove prefix to get only generated text
            if response.startswith(prefix):
                generated_text = response[len(prefix):].strip()
            else:
                # Fallback: just use the response as-is if prefix doesn't match
                generated_text = response
        
        record = ExperimentRecord(
            timestamp=datetime.now().isoformat(),
            experiment_id=f"{self.experiment_name}_{len(self.records)}",
            model_name=model_name,
            prompt=prompt,
            attack_type=attack_type,
            suffix=suffix,
            response=response[:500] if response else None,  # Full response (truncated)
            generated_text=generated_text[:500] if generated_text else None,  # Only generated part
            success=success,
            metrics=metrics,
            layer_data=layer_data,
            attention_data=attention_data,
        )
        
        self.records.append(record)
        self.metrics_history.append({
            "step": len(self.records),
            **metrics,
        })
    
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log metrics at a given step."""
        self.metrics_history.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        })
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save records to CSV file.
        
        Args:
            filename: Output filename (default: records.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "records.csv"
        
        filepath = self.data_dir / filename
        
        if not self.records:
            return str(filepath)
        
        # Flatten records for CSV
        rows = []
        for record in self.records:
            row = {
                "timestamp": record.timestamp,
                "experiment_id": record.experiment_id,
                "model_name": record.model_name,
                "prompt": record.prompt[:100] if record.prompt else "",
                "attack_type": record.attack_type,
                "suffix": record.suffix[:50] if record.suffix else "",
                "success": record.success,
            }
            # Add metrics
            for key, value in record.metrics.items():
                row[f"metric_{key}"] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Save records to JSON file.
        
        Args:
            filename: Output filename (default: records.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "records.json"
        
        filepath = self.data_dir / filename
        
        data = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "records": [asdict(r) for r in self.records],
            "metrics_history": self.metrics_history,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_metrics_history(self, filename: Optional[str] = None) -> str:
        """Save metrics history to CSV."""
        if filename is None:
            filename = "metrics_history.csv"
        
        filepath = self.data_dir / filename
        
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics."""
        if not self.records:
            return {"total_attacks": 0}
        
        total = len(self.records)
        successful = sum(1 for r in self.records if r.success)
        
        # Aggregate metrics
        all_metrics = {}
        for record in self.records:
            for key, value in record.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        
        return {
            "experiment_name": self.experiment_name,
            "total_attacks": total,
            "successful_attacks": successful,
            "attack_success_rate": successful / total if total > 0 else 0,
            "average_metrics": avg_metrics,
            "output_directory": str(self.output_dir / self.experiment_name),
        }
    
    def get_charts_directory(self) -> str:
        """Get path to charts directory."""
        return str(self.charts_dir)
