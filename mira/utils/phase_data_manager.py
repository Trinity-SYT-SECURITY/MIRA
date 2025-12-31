"""
Phase Data Manager - Records and saves data for each analysis phase.

Ensures all phases have complete data records, charts, and metrics
for comprehensive reporting and analysis.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class PhaseData:
    """Data structure for a single phase."""
    phase_number: int
    phase_name: str
    phase_detail: str
    timestamp: str
    duration: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    charts: List[str] = field(default_factory=list)  # Chart file paths
    data_files: List[str] = field(default_factory=list)  # Data file paths
    raw_data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class PhaseDataManager:
    """
    Manages data recording and saving for each analysis phase.
    
    Records:
    - Phase metrics and statistics
    - Generated charts
    - Raw data files (JSON, CSV)
    - Phase summaries
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize phase data manager.
        
        Args:
            output_dir: Base output directory for all phase data
        """
        self.output_dir = Path(output_dir)
        self.phases_dir = self.output_dir / "phases"
        self.phases_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.phases_dir / "data"
        self.charts_dir = self.phases_dir / "charts"
        self.data_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        
        # Store phase data
        self.phases: Dict[int, PhaseData] = {}
        self.current_phase: Optional[int] = None
        self.phase_start_time: Optional[float] = None
    
    def start_phase(
        self,
        phase_number: int,
        phase_name: str,
        phase_detail: str = "",
    ) -> None:
        """
        Start recording data for a phase.
        
        Args:
            phase_number: Phase number (0-7)
            phase_name: Name of the phase
            phase_detail: Optional detail string
        """
        from time import time
        
        # End previous phase if exists
        if self.current_phase is not None:
            self.end_phase()
        
        self.current_phase = phase_number
        self.phase_start_time = time()
        
        # Create phase data
        phase_data = PhaseData(
            phase_number=phase_number,
            phase_name=phase_name,
            phase_detail=phase_detail,
            timestamp=datetime.now().isoformat(),
        )
        
        self.phases[phase_number] = phase_data
    
    def end_phase(self) -> None:
        """End current phase and calculate duration."""
        from time import time
        
        if self.current_phase is None:
            return
        
        if self.phase_start_time is not None:
            duration = time() - self.phase_start_time
            self.phases[self.current_phase].duration = duration
        
        self.current_phase = None
        self.phase_start_time = None
    
    def record_metric(self, key: str, value: Any) -> None:
        """
        Record a metric for the current phase.
        
        Args:
            key: Metric name
            value: Metric value
        """
        if self.current_phase is not None:
            self.phases[self.current_phase].metrics[key] = value
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        if self.current_phase is not None:
            self.phases[self.current_phase].metrics.update(metrics)
    
    def register_chart(self, chart_path: str) -> None:
        """
        Register a chart file for the current phase.
        
        Args:
            chart_path: Path to the chart file
        """
        if self.current_phase is not None:
            # Convert to relative path if absolute
            chart_path_str = str(chart_path)
            if Path(chart_path_str).is_absolute():
                try:
                    chart_path_str = str(Path(chart_path_str).relative_to(self.output_dir))
                except ValueError:
                    pass  # Keep absolute path if can't make relative
            self.phases[self.current_phase].charts.append(chart_path_str)
    
    def save_phase_data(
        self,
        phase_number: int,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save raw data for a phase to JSON file.
        
        Args:
            phase_number: Phase number
            data: Data dictionary to save
            filename: Optional custom filename (default: phase_{number}_data.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"phase_{phase_number}_data.json"
        
        file_path = self.data_dir / filename
        
        # Add metadata
        phase = self.phases.get(phase_number)
        phase_name = phase.phase_name if phase else f"Phase {phase_number}"
        
        data_with_meta = {
            "phase_number": phase_number,
            "phase_name": phase_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_with_meta, f, indent=2, ensure_ascii=False)
        
        # Register file
        if phase_number in self.phases:
            self.phases[phase_number].data_files.append(str(file_path.relative_to(self.output_dir)))
        
        return file_path
    
    def save_phase_csv(
        self,
        phase_number: int,
        data: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save phase data to CSV file.
        
        Args:
            phase_number: Phase number
            data: List of dictionaries (rows)
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"phase_{phase_number}_data.csv"
        
        file_path = self.data_dir / filename
        
        if not data:
            # Create empty CSV with headers from phase metrics
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("phase_number,phase_name,timestamp\n")
            return file_path
        
        # Write CSV
        fieldnames = list(data[0].keys()) if data else []
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        # Register file
        if phase_number in self.phases:
            self.phases[phase_number].data_files.append(str(file_path.relative_to(self.output_dir)))
        
        return file_path
    
    def set_phase_summary(self, phase_number: int, summary: str) -> None:
        """
        Set a text summary for a phase.
        
        Args:
            phase_number: Phase number
            summary: Summary text
        """
        if phase_number in self.phases:
            self.phases[phase_number].summary = summary
    
    def set_phase_raw_data(self, phase_number: int, raw_data: Dict[str, Any]) -> None:
        """
        Store raw data for a phase (for later saving).
        
        Args:
            phase_number: Phase number
            raw_data: Raw data dictionary
        """
        if phase_number in self.phases:
            self.phases[phase_number].raw_data.update(raw_data)
    
    def save_all_phases(self) -> Dict[str, Any]:
        """
        Save all phase data to JSON and create summary.
        
        Returns:
            Dictionary with all phase data
        """
        # Save individual phase files
        all_phases_data = {}
        for phase_num, phase_data in sorted(self.phases.items()):
            # Save phase JSON
            phase_file = self.data_dir / f"phase_{phase_num}_summary.json"
            with open(phase_file, "w", encoding="utf-8") as f:
                json.dump(asdict(phase_data), f, indent=2, ensure_ascii=False)
            
            # Save raw data if exists
            if phase_data.raw_data:
                raw_data_file = self.data_dir / f"phase_{phase_num}_raw_data.json"
                with open(raw_data_file, "w", encoding="utf-8") as f:
                    json.dump(phase_data.raw_data, f, indent=2, ensure_ascii=False)
            
            all_phases_data[f"phase_{phase_num}"] = asdict(phase_data)
        
        # Save combined phases summary
        summary_file = self.phases_dir / "all_phases_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_phases_data, f, indent=2, ensure_ascii=False)
        
        # Create CSV summary
        csv_file = self.phases_dir / "phases_summary.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "phase_number", "phase_name", "phase_detail", "timestamp",
                "duration", "num_charts", "num_data_files", "num_metrics"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for phase_num, phase_data in sorted(self.phases.items()):
                writer.writerow({
                    "phase_number": phase_data.phase_number,
                    "phase_name": phase_data.phase_name,
                    "phase_detail": phase_data.phase_detail,
                    "timestamp": phase_data.timestamp,
                    "duration": phase_data.duration,
                    "num_charts": len(phase_data.charts),
                    "num_data_files": len(phase_data.data_files),
                    "num_metrics": len(phase_data.metrics),
                })
        
        return all_phases_data
    
    def get_phase_data(self, phase_number: int) -> Optional[PhaseData]:
        """Get phase data for a specific phase."""
        return self.phases.get(phase_number)
    
    def get_all_phases(self) -> Dict[int, PhaseData]:
        """Get all phase data."""
        return self.phases.copy()

