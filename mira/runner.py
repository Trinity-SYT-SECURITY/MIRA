"""
Integrated experiment runner with auto visualization.

Runs complete experiment pipelines with automatic:
- Environment detection
- Data logging
- Chart generation
- Summary reports
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from mira.utils.environment import detect_environment, print_environment_info
from mira.utils.experiment_logger import ExperimentLogger
from mira.visualization.research_charts import ResearchChartGenerator
from mira.config import load_config, MiraConfig


class ExperimentRunner:
    """
    Complete experiment runner with integrated visualization.
    
    Handles the full research workflow:
    1. Detect environment and configure settings
    2. Load model appropriate for hardware
    3. Run experiments with logging
    4. Generate charts automatically
    5. Save structured results
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        output_dir: str = "./results",
        config_path: Optional[str] = None,
        auto_detect_env: bool = True,
    ):
        """
        Initialize experiment runner.
        
        Args:
            experiment_name: Name for this experiment
            output_dir: Directory for results
            config_path: Path to config file
            auto_detect_env: Auto-detect hardware on init
        """
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Load configuration
        self.config = MiraConfig.load(config_path)
        
        # Detect environment
        self.env = None
        if auto_detect_env:
            self.env = detect_environment()
            self._apply_env_recommendations()
        
        # Initialize logger
        self.logger = ExperimentLogger(
            output_dir=str(self.output_dir),
            experiment_name=experiment_name,
        )
        
        # Initialize chart generator
        self.chart_gen = ResearchChartGenerator(
            output_dir=self.logger.get_charts_directory(),
        )
        
        # Model (loaded lazily)
        self._model = None
    
    def _apply_env_recommendations(self) -> None:
        """Apply environment-based recommendations to config."""
        if self.env is None:
            return
        
        # Update model config based on hardware
        if not self.env.gpu.available:
            # CPU mode - use smaller model
            self.config.model.name = self.env.recommended_model
        
        self.config.model.dtype = self.env.recommended_dtype
        self.config.analysis.batch_size = min(
            self.config.analysis.batch_size,
            self.env.max_batch_size,
        )
    
    def print_environment(self) -> None:
        """Print environment information."""
        if self.env:
            print_environment_info(self.env)
        else:
            self.env = detect_environment()
            print_environment_info(self.env)
    
    def load_model(self, model_name: Optional[str] = None):
        """
        Load model with environment-appropriate settings.
        
        Args:
            model_name: Override model name from config
        """
        from mira.core import ModelWrapper
        
        name = model_name or self.config.model.name
        device = self.config.model.get_device()
        
        print(f"Loading model: {name}")
        print(f"Device: {device}")
        
        self._model = ModelWrapper(name, device=device)
        
        return self._model
    
    def run_subspace_analysis(
        self,
        safe_prompts: List[str],
        harmful_prompts: List[str],
        layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run subspace analysis with auto visualization.
        
        Args:
            safe_prompts: List of safe prompts
            harmful_prompts: List of harmful prompts
            layer_idx: Layer to analyze
            
        Returns:
            Analysis results
        """
        if self._model is None:
            self.load_model()
        
        from mira.analysis import SubspaceAnalyzer
        
        layer = layer_idx or (self._model.n_layers // 2)
        analyzer = SubspaceAnalyzer(self._model, layer_idx=layer)
        
        # Collect embeddings
        print("Collecting activations...")
        safe_embeds = analyzer.collect_activations(safe_prompts)
        unsafe_embeds = analyzer.collect_activations(harmful_prompts)
        
        # Train probe
        print("Training linear probe...")
        result = analyzer.train_probe(safe_prompts, harmful_prompts)
        
        # Generate visualization
        from mira.visualization import plot_subspace_2d
        
        chart_path = plot_subspace_2d(
            safe_embeds,
            unsafe_embeds,
            refusal_direction=result.refusal_direction,
            title=f"Subspace Analysis (Layer {layer})",
            save_path=str(Path(self.logger.get_charts_directory()) / "subspace_analysis.png"),
        )
        
        # Log metrics
        self.logger.log_metrics(0, {
            "probe_accuracy": result.probe_accuracy,
            "layer": layer,
            "n_safe_prompts": len(safe_prompts),
            "n_harmful_prompts": len(harmful_prompts),
        })
        
        return {
            "probe_accuracy": result.probe_accuracy,
            "layer": layer,
            "refusal_direction_norm": float(result.refusal_direction.norm()),
            "chart_path": chart_path,
        }
    
    def run_attack(
        self,
        prompt: str,
        attack_type: str = "gradient",
        num_steps: int = 50,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run attack with auto logging and visualization.
        
        Args:
            prompt: Target prompt
            attack_type: "gradient", "rerouting", or "proxy"
            num_steps: Optimization steps
            
        Returns:
            Attack results
        """
        if self._model is None:
            self.load_model()
        
        # Select attack class
        if attack_type == "gradient":
            from mira.attack import GradientAttack
            attack = GradientAttack(self._model, **kwargs)
        elif attack_type == "rerouting":
            from mira.attack import ReroutingAttack
            attack = ReroutingAttack(self._model, **kwargs)
        else:
            from mira.attack import GradientAttack
            attack = GradientAttack(self._model, **kwargs)
        
        print(f"Running {attack_type} attack...")
        result = attack.optimize(prompt, num_steps=num_steps, verbose=True)
        
        # Generate loss curve
        if result.loss_history:
            self.chart_gen.plot_loss_curve(
                result.loss_history,
                title=f"{attack_type.title()} Attack Optimization",
                save_name=f"attack_loss_{attack_type}",
            )
        
        # Log attack
        self.logger.log_attack(
            model_name=self._model.model_name,
            prompt=prompt,
            attack_type=attack_type,
            suffix=result.adversarial_suffix,
            response=result.generated_response or "",
            success=result.success,
            metrics={
                "final_loss": result.final_loss,
                "num_steps": result.num_steps,
            },
        )
        
        return {
            "success": result.success,
            "suffix": result.adversarial_suffix,
            "final_loss": result.final_loss,
            "response_preview": (result.generated_response or "")[:200],
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate experiment summary with all charts.
        
        Returns:
            Summary dictionary
        """
        summary = self.logger.get_summary()
        
        # Save data
        csv_path = self.logger.save_to_csv()
        json_path = self.logger.save_to_json()
        metrics_path = self.logger.save_metrics_history()
        
        summary["data_files"] = {
            "csv": csv_path,
            "json": json_path,
            "metrics": metrics_path,
        }
        summary["charts_directory"] = self.logger.get_charts_directory()
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Name: {summary.get('experiment_name', 'N/A')}")
        print(f"Total Attacks: {summary.get('total_attacks', 0)}")
        print(f"Successful: {summary.get('successful_attacks', 0)}")
        print(f"ASR: {summary.get('attack_success_rate', 0):.2%}")
        print(f"Output: {summary.get('output_directory', 'N/A')}")
        print("=" * 60)
        
        return summary


def run_quick_experiment(
    prompts: List[str],
    model_name: Optional[str] = None,
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """
    Quick experiment runner for testing.
    
    Args:
        prompts: List of prompts to test
        model_name: Model to use
        output_dir: Output directory
        
    Returns:
        Experiment results
    """
    runner = ExperimentRunner(output_dir=output_dir)
    runner.print_environment()
    
    if model_name:
        runner.load_model(model_name)
    else:
        runner.load_model()
    
    results = []
    for prompt in prompts:
        result = runner.run_attack(prompt)
        results.append(result)
    
    summary = runner.generate_summary()
    summary["individual_results"] = results
    
    return summary
