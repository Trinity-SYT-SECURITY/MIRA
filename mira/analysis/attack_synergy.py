"""
Attack Type Synergy Analysis.

Analyzes synergistic effects between different attack types:
- Prompt-based vs Gradient-based interactions
- SSR enhancement effects on other attacks
- Combined attack strategies
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class SynergyMetrics:
    """Metrics for attack type synergy."""
    attack_type_1: str
    attack_type_2: str
    individual_asr_1: float = 0.0
    individual_asr_2: float = 0.0
    combined_asr: float = 0.0
    synergy_score: float = 0.0
    enhancement_factor: float = 1.0


class AttackSynergyAnalyzer:
    """Analyze synergistic effects between attack types."""
    
    def __init__(self):
        """Initialize synergy analyzer."""
        self.attack_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def compute_synergy_score(
        self,
        asr_1: float,
        asr_2: float,
        combined_asr: float,
    ) -> float:
        """
        Compute synergy score between two attack types.
        
        Synergy = (combined_asr - max(asr_1, asr_2)) / (1 - max(asr_1, asr_2))
        Positive values indicate synergy, negative indicate interference.
        
        Args:
            asr_1: ASR of first attack type alone
            asr_2: ASR of second attack type alone
            combined_asr: ASR when both attacks are combined
            
        Returns:
            Synergy score (-1 to 1)
        """
        max_individual = max(asr_1, asr_2)
        
        if max_individual >= 1.0:
            return 0.0  # Can't improve beyond 100%
        
        if combined_asr <= max_individual:
            # No synergy or interference
            return (combined_asr - max_individual) / (1.0 - max_individual + 1e-10)
        
        # Positive synergy
        improvement = combined_asr - max_individual
        max_possible_improvement = 1.0 - max_individual
        
        return improvement / (max_possible_improvement + 1e-10)
    
    def analyze_prompt_gradient_synergy(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze synergy between prompt-based and gradient-based attacks.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            Dict with synergy analysis results
        """
        prompt_attacks = []
        gradient_attacks = []
        combined_attacks = []
        
        for result in all_results:
            if not result.get("success", False):
                continue
            
            attack_details = result.get("attack_details", [])
            
            for detail in attack_details:
                attack_type = detail.get("attack_type", "")
                success = detail.get("success", False)
                
                if attack_type in ["dan", "roleplay", "social", "logic", "encoding"]:
                    prompt_attacks.append(success)
                elif attack_type == "gradient":
                    gradient_attacks.append(success)
                elif "ssr" in attack_type.lower():
                    # SSR can be combined with others
                    combined_attacks.append(success)
        
        # Compute individual ASRs
        prompt_asr = np.mean(prompt_attacks) if prompt_attacks else 0.0
        gradient_asr = np.mean(gradient_attacks) if gradient_attacks else 0.0
        
        # For combined attacks, we need to check if they were used together
        # This is a simplified analysis - in practice, we'd track which attacks were combined
        combined_asr = np.mean(combined_attacks) if combined_attacks else max(prompt_asr, gradient_asr)
        
        synergy_score = self.compute_synergy_score(prompt_asr, gradient_asr, combined_asr)
        enhancement_factor = combined_asr / max(prompt_asr, gradient_asr) if max(prompt_asr, gradient_asr) > 0 else 1.0
        
        return {
            "prompt_asr": float(prompt_asr),
            "gradient_asr": float(gradient_asr),
            "combined_asr": float(combined_asr),
            "synergy_score": float(synergy_score),
            "enhancement_factor": float(enhancement_factor),
            "num_prompt_attacks": len(prompt_attacks),
            "num_gradient_attacks": len(gradient_attacks),
        }
    
    def analyze_ssr_enhancement(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze how SSR attacks enhance other attack types.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            Dict with SSR enhancement analysis
        """
        ssr_attacks = []
        non_ssr_attacks = []
        ssr_enhanced_attacks = []
        
        for result in all_results:
            if not result.get("success", False):
                continue
            
            attack_details = result.get("attack_details", [])
            has_ssr = False
            has_other = False
            
            for detail in attack_details:
                attack_type = detail.get("attack_type", "")
                success = detail.get("success", False)
                
                if "ssr" in attack_type.lower():
                    has_ssr = True
                    ssr_attacks.append(success)
                else:
                    has_other = True
                    non_ssr_attacks.append(success)
            
            # If both SSR and other attacks present, consider it enhanced
            if has_ssr and has_other:
                for detail in attack_details:
                    if "ssr" not in detail.get("attack_type", "").lower():
                        ssr_enhanced_attacks.append(detail.get("success", False))
        
        ssr_asr = np.mean(ssr_attacks) if ssr_attacks else 0.0
        non_ssr_asr = np.mean(non_ssr_attacks) if non_ssr_attacks else 0.0
        enhanced_asr = np.mean(ssr_enhanced_attacks) if ssr_enhanced_attacks else non_ssr_asr
        
        enhancement_factor = enhanced_asr / non_ssr_asr if non_ssr_asr > 0 else 1.0
        
        return {
            "ssr_asr": float(ssr_asr),
            "non_ssr_asr": float(non_ssr_asr),
            "enhanced_asr": float(enhanced_asr),
            "enhancement_factor": float(enhancement_factor),
            "num_ssr_attacks": len(ssr_attacks),
            "num_enhanced_attacks": len(ssr_enhanced_attacks),
        }
    
    def analyze_attack_combinations(
        self,
        all_results: List[Dict[str, Any]],
    ) -> Dict[str, SynergyMetrics]:
        """
        Analyze all possible attack type combinations.
        
        Args:
            all_results: Complete results from all models
            
        Returns:
            Dict mapping (attack_type_1, attack_type_2) to SynergyMetrics
        """
        # Group attacks by type
        attack_groups = defaultdict(lambda: {"success": 0, "total": 0})
        combined_groups = defaultdict(lambda: {"success": 0, "total": 0})
        
        for result in all_results:
            if not result.get("success", False):
                continue
            
            attack_details = result.get("attack_details", [])
            attack_types = set()
            
            for detail in attack_details:
                attack_type = detail.get("attack_type", "unknown")
                attack_types.add(attack_type)
                
                # Individual attack stats
                attack_groups[attack_type]["total"] += 1
                if detail.get("success", False):
                    attack_groups[attack_type]["success"] += 1
            
            # Combined attack stats (if multiple types in same result)
            if len(attack_types) > 1:
                types_tuple = tuple(sorted(attack_types))
                combined_groups[types_tuple]["total"] += 1
                
                # Success if any attack succeeded
                if any(d.get("success", False) for d in attack_details):
                    combined_groups[types_tuple]["success"] += 1
        
        # Compute synergy metrics
        synergy_metrics = {}
        
        for (type1, type2), combined_data in combined_groups.items():
            if combined_data["total"] == 0:
                continue
            
            combined_asr = combined_data["success"] / combined_data["total"]
            
            # Get individual ASRs
            asr_1 = attack_groups[type1]["success"] / attack_groups[type1]["total"] if attack_groups[type1]["total"] > 0 else 0.0
            asr_2 = attack_groups[type2]["success"] / attack_groups[type2]["total"] if attack_groups[type2]["total"] > 0 else 0.0
            
            synergy_score = self.compute_synergy_score(asr_1, asr_2, combined_asr)
            enhancement_factor = combined_asr / max(asr_1, asr_2) if max(asr_1, asr_2) > 0 else 1.0
            
            key = f"{type1}+{type2}"
            synergy_metrics[key] = SynergyMetrics(
                attack_type_1=type1,
                attack_type_2=type2,
                individual_asr_1=asr_1,
                individual_asr_2=asr_2,
                combined_asr=combined_asr,
                synergy_score=synergy_score,
                enhancement_factor=enhancement_factor,
            )
        
        return synergy_metrics

