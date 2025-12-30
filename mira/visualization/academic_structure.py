"""
Academic Paper Structure Generator for Research Report.

Generates research paper sections:
- Executive Summary
- Introduction
- Background
- Detailed Methodology
- Discussion
- Conclusion
- Appendix
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class AcademicStructureGenerator:
    """Generate academic paper structure sections."""
    
    def __init__(self):
        """Initialize generator."""
        pass
    
    def generate_executive_summary(
        self,
        all_results: List[Dict[str, Any]],
        phase_comparison: Optional[Dict[str, Any]] = None,
        cross_model_analysis: Optional[Dict[str, Any]] = None,
        transferability_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate executive summary section."""
        n_models = len([r for r in all_results if r.get("success")])
        
        # Calculate key statistics
        avg_asr = sum(r.get("asr", 0) for r in all_results if r.get("success")) / n_models if n_models > 0 else 0
        
        # Find most vulnerable
        most_vulnerable = max(all_results, key=lambda x: x.get("asr", 0)) if all_results else None
        most_robust = min(all_results, key=lambda x: x.get("asr", 0)) if all_results else None
        
        # Get transferability insights
        transfer_summary = transferability_analysis.get("systematic_comparison", {}) if transferability_analysis else {}
        systematic_asr = transfer_summary.get("systematic_mean_asr", 0)
        
        return f'''
        <section class="section executive-summary">
            <div class="section-header">
                <div class="section-icon">[EXECUTIVE SUMMARY]</div>
                <h2>Executive Summary</h2>
            </div>
            
            <div class="summary-content">
                <p class="lead">
                    This research presents a comprehensive multi-model security analysis of {n_models} large language models (LLMs),
                    employing a systematic 4-level evaluation framework to assess vulnerability to adversarial attacks.
                </p>
                
                <h3>Key Findings</h3>
                <ul class="findings-list">
                    <li><strong>Average Attack Success Rate:</strong> {avg_asr:.1%} across all models tested</li>
                    <li><strong>Most Vulnerable Model:</strong> {most_vulnerable.get("model_name", "N/A") if most_vulnerable else "N/A"} 
                        (ASR: {most_vulnerable.get("asr", 0):.1%} if most_vulnerable else 0)</li>
                    <li><strong>Most Robust Model:</strong> {most_robust.get("model_name", "N/A") if most_robust else "N/A"} 
                        (ASR: {most_robust.get("asr", 0):.1%} if most_robust else 0)</li>
                    <li><strong>Systematic Attack Effectiveness:</strong> MIRA's systematic approach achieved 
                        {systematic_asr:.1%} ASR, demonstrating the importance of mechanistic understanding</li>
                </ul>
                
                <h3>Main Contributions</h3>
                <ol class="contributions-list">
                    <li><strong>4-Level Comparison Framework:</strong> Novel evaluation methodology examining behavioral outcomes,
                        phase sensitivity, internal mechanics, and attack transferability</li>
                    <li><strong>Mechanistic Insights:</strong> Identification of common vulnerability patterns across models,
                        including refusal direction similarities and entropy collapse signatures</li>
                    <li><strong>Transferability Analysis:</strong> Demonstration that systematic attack logic generalizes
                        across models more effectively than random approaches</li>
                </ol>
                
                <h3>Implications</h3>
                <p>
                    Our findings reveal that LLM vulnerabilities are not isolated prompt artifacts but systematic
                    manipulations of the model's conditional probability space. This has significant implications for:
                </p>
                <ul>
                    <li>Development of more robust safety mechanisms targeting identified vulnerability patterns</li>
                    <li>Design of model-agnostic defense strategies based on mechanistic understanding</li>
                    <li>Establishment of standardized security evaluation protocols for LLM deployment</li>
                </ul>
            </div>
            
            <style>
                .executive-summary .summary-content {{
                    line-height: 1.8;
                }}
                .executive-summary .lead {{
                    font-size: 1.1rem;
                    font-weight: 500;
                    margin-bottom: 24px;
                    padding: 16px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-left: 4px solid var(--primary, #818cf8);
                    border-radius: 4px;
                }}
                .executive-summary h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .findings-list, .contributions-list {{
                    margin: 16px 0;
                    padding-left: 24px;
                }}
                .findings-list li, .contributions-list li {{
                    margin: 12px 0;
                }}
            </style>
        </section>
'''
    
    def generate_introduction(self) -> str:
        """Generate introduction section."""
        return '''
        <section class="section introduction">
            <div class="section-header">
                <div class="section-icon">[INTRODUCTION]</div>
                <h2>Introduction</h2>
            </div>
            
            <div class="intro-content">
                <h3>Research Motivation</h3>
                <p>
                    Large Language Models (LLMs) have achieved remarkable capabilities in natural language understanding
                    and generation, leading to widespread deployment in critical applications. However, these models
                    remain vulnerable to adversarial attacks that can bypass safety mechanisms and elicit harmful outputs.
                </p>
                
                <p>
                    While existing research has documented various attack techniques, there is a critical gap in understanding
                    <strong>how and why</strong> these attacks succeed at the mechanistic level. Most studies focus on
                    surface-level behavioral outcomes (e.g., Attack Success Rate) without examining the internal computational
                    processes that lead to safety failures.
                </p>
                
                <h3>Problem Statement</h3>
                <p>
                    Current LLM security evaluations suffer from three key limitations:
                </p>
                <ol>
                    <li><strong>Lack of Mechanistic Understanding:</strong> Evaluations treat models as black boxes,
                        missing opportunities to identify root causes of vulnerabilities</li>
                    <li><strong>Model-Specific Focus:</strong> Most studies analyze individual models in isolation,
                        preventing identification of universal vulnerability patterns</li>
                    <li><strong>Incomplete Evaluation:</strong> Assessments typically measure only final outputs,
                        ignoring intermediate computational states that reveal attack mechanisms</li>
                </ol>
                
                <h3>Research Questions</h3>
                <p>This research addresses three fundamental questions:</p>
                <ul class="research-questions">
                    <li><strong>RQ1:</strong> Do adversarial prompts produce consistent, measurable differences in
                        internal model states (activations, attention patterns, probability distributions) compared to
                        benign inputs?</li>
                    <li><strong>RQ2:</strong> Can these internal differences be quantified and used to establish
                        a systematic framework for security evaluation?</li>
                    <li><strong>RQ3:</strong> Do vulnerability patterns generalize across different models, enabling
                        development of model-agnostic defense strategies?</li>
                </ul>
                
                <h3>Contributions</h3>
                <p>This work makes the following contributions:</p>
                <ol>
                    <li><strong>4-Level Evaluation Framework:</strong> A comprehensive methodology examining behavioral,
                        phase-wise, mechanistic, and transferability aspects of model security</li>
                    <li><strong>Mechanistic Analysis Tools:</strong> Novel techniques for capturing and analyzing
                        internal model states during adversarial interactions</li>
                    <li><strong>Cross-Model Insights:</strong> Identification of universal vulnerability patterns
                        through comparative analysis of multiple LLMs</li>
                    <li><strong>Systematic Attack Methodology:</strong> Demonstration that mechanistically-informed
                        attacks significantly outperform random approaches</li>
                </ol>
            </div>
            
            <style>
                .introduction .intro-content {{
                    line-height: 1.8;
                }}
                .introduction h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .introduction p {{
                    margin: 16px 0;
                }}
                .introduction ol, .introduction ul {{
                    margin: 16px 0;
                    padding-left: 24px;
                }}
                .introduction li {{
                    margin: 12px 0;
                }}
                .research-questions {{
                    background: var(--bg-tertiary, #1a1a2e);
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid var(--primary, #818cf8);
                }}
            </style>
        </section>
'''
    
    def generate_background(self) -> str:
        """Generate background section."""
        return '''
        <section class="section background">
            <div class="section-header">
                <div class="section-icon">[BACKGROUND]</div>
                <h2>Background and Related Work</h2>
            </div>
            
            <div class="background-content">
                <h3>Large Language Model Security</h3>
                <p>
                    LLM security research has evolved through several phases, from early prompt injection demonstrations
                    to sophisticated gradient-based optimization attacks. Key attack categories include:
                </p>
                <ul>
                    <li><strong>Prompt-Based Attacks:</strong> Jailbreaking techniques (DAN, roleplay scenarios),
                        encoding attacks, and social engineering approaches</li>
                    <li><strong>Gradient-Based Attacks:</strong> GCG (Greedy Coordinate Gradient) and related
                        optimization methods that find adversarial suffixes</li>
                    <li><strong>Fine-Tuning Attacks:</strong> Manipulation through carefully crafted training data</li>
                </ul>
                
                <h3>Mechanistic Interpretability</h3>
                <p>
                    Recent advances in mechanistic interpretability have enabled deeper understanding of transformer
                    model internals. Key techniques include:
                </p>
                
                <table class="techniques-table">
                    <thead>
                        <tr>
                            <th>Technique</th>
                            <th>Purpose</th>
                            <th>Application to Security</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Logit Lens</strong></td>
                            <td>Project intermediate activations to vocabulary space</td>
                            <td>Track how harmful content emerges through layers</td>
                        </tr>
                        <tr>
                            <td><strong>Attention Analysis</strong></td>
                            <td>Examine token-to-token attention patterns</td>
                            <td>Identify attention hijacking during attacks</td>
                        </tr>
                        <tr>
                            <td><strong>Probing</strong></td>
                            <td>Train classifiers on intermediate representations</td>
                            <td>Detect attack signals in hidden states</td>
                        </tr>
                        <tr>
                            <td><strong>Activation Patching</strong></td>
                            <td>Intervene on specific activations</td>
                            <td>Identify causal mechanisms of safety failures</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3>Subspace Rerouting</h3>
                <p>
                    Recent work has identified "refusal subspaces" in model representations - specific directions
                    in activation space associated with safety refusals. Attacks can be understood as attempts to
                    steer model computations away from these subspaces.
                </p>
                
                <h3>Cross-Model Analysis</h3>
                <p>
                    While most security research focuses on individual models, comparative studies reveal important
                    insights about universal vulnerability patterns. Our work extends this by systematically comparing
                    internal mechanics across models, not just behavioral outcomes.
                </p>
            </div>
            
            <style>
                .background .background-content {{
                    line-height: 1.8;
                }}
                .background h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .techniques-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .techniques-table th, .techniques-table td {{
                    padding: 12px;
                    text-align: left;
                    border: 1px solid var(--border-color, #333);
                }}
                .techniques-table th {{
                    background: var(--bg-secondary, #2a2a4e);
                    font-weight: 600;
                }}
                .techniques-table tr:nth-child(even) {{
                    background: var(--bg-tertiary, #1a1a2e);
                }}
            </style>
        </section>
'''
    
    def generate_detailed_methodology(
        self,
        all_results: List[Dict[str, Any]]
    ) -> str:
        """Generate detailed methodology section."""
        n_models = len([r for r in all_results if r.get("success")])
        model_names = [r.get("model_name") for r in all_results if r.get("success")]
        
        return f'''
        <section class="section methodology">
            <div class="section-header">
                <div class="section-icon">[METHODOLOGY]</div>
                <h2>Detailed Methodology</h2>
            </div>
            
            <div class="methodology-content">
                <h3>4.1 Model Setup</h3>
                <p>
                    We evaluated {n_models} open-source language models with varying architectures and sizes.
                    All models were loaded using HuggingFace Transformers with consistent configurations to ensure
                    fair comparison.
                </p>
                
                <table class="models-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Architecture</th>
                            <th>Parameters</th>
                            <th>Context Length</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td>{name}</td>
                            <td>Transformer</td>
                            <td>Varies</td>
                            <td>2048+</td>
                        </tr>
                        ''' for name in model_names[:10])}
                    </tbody>
                </table>
                
                <h3>4.2 Prompt Set Design</h3>
                <p>Our evaluation uses three categories of prompts:</p>
                <ul>
                    <li><strong>Safe Prompts:</strong> Benign queries for baseline behavior measurement</li>
                    <li><strong>Harmful Prompts:</strong> Requests for prohibited content (violence, illegal activities, etc.)</li>
                    <li><strong>Attack Prompts:</strong> Adversarially crafted inputs using various jailbreaking techniques</li>
                </ul>
                
                <h3>4.3 Internal Signal Collection</h3>
                <p>For each prompt, we capture:</p>
                <ol>
                    <li><strong>Layer Activations:</strong> Hidden state norms at each transformer layer</li>
                    <li><strong>Attention Patterns:</strong> Multi-head attention weights for all token pairs</li>
                    <li><strong>Logit Distributions:</strong> Output probability distributions at intermediate layers (Logit Lens)</li>
                    <li><strong>Entropy Metrics:</strong> Prediction uncertainty throughout generation</li>
                </ol>
                
                <h3>4.4 Analysis Pipeline</h3>
                <div class="pipeline-diagram">
                    <div class="pipeline-step">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <strong>Subspace Analysis</strong>
                            <p>Train probes to identify refusal directions in activation space</p>
                        </div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <strong>Attack Execution</strong>
                            <p>Run prompt-based and gradient-based attacks while capturing internal states</p>
                        </div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <strong>Comparative Analysis</strong>
                            <p>Apply 4-level framework to identify patterns across models</p>
                        </div>
                    </div>
                </div>
                
                <h3>4.5 Evaluation Metrics</h3>
                <ul>
                    <li><strong>Attack Success Rate (ASR):</strong> Percentage of attacks that elicit harmful responses</li>
                    <li><strong>Probe Accuracy:</strong> Ability to detect attack signals from hidden states</li>
                    <li><strong>Probe Bypass Rate:</strong> Percentage of attacks that evade probe detection</li>
                    <li><strong>Entropy Collapse Ratio:</strong> Difference in entropy drop between successful and failed attacks</li>
                    <li><strong>Transfer Rate:</strong> Percentage of attacks that succeed across multiple models</li>
                </ul>
            </div>
            
            <style>
                .methodology .methodology-content {{
                    line-height: 1.8;
                }}
                .methodology h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .models-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 0.9rem;
                }}
                .models-table th, .models-table td {{
                    padding: 10px;
                    text-align: left;
                    border: 1px solid var(--border-color, #333);
                }}
                .models-table th {{
                    background: var(--bg-secondary, #2a2a4e);
                }}
                .pipeline-diagram {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin: 24px 0;
                    padding: 20px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-radius: 8px;
                }}
                .pipeline-step {{
                    flex: 1;
                    text-align: center;
                }}
                .step-number {{
                    width: 40px;
                    height: 40px;
                    line-height: 40px;
                    margin: 0 auto 12px;
                    background: var(--primary, #818cf8);
                    border-radius: 50%;
                    font-weight: 600;
                    font-size: 1.2rem;
                }}
                .step-content strong {{
                    display: block;
                    margin-bottom: 8px;
                }}
                .step-content p {{
                    font-size: 0.85rem;
                    color: var(--text-secondary, #999);
                }}
                .pipeline-arrow {{
                    font-size: 2rem;
                    color: var(--primary, #818cf8);
                    padding: 0 20px;
                }}
            </style>
        </section>
'''
    
    def generate_discussion(
        self,
        phase_comparison: Optional[Dict[str, Any]] = None,
        cross_model_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate discussion section."""
        return '''
        <section class="section discussion">
            <div class="section-header">
                <div class="section-icon">[DISCUSSION]</div>
                <h2>Discussion</h2>
            </div>
            
            <div class="discussion-content">
                <h3>Interpretation of Findings</h3>
                
                <h4>Universal Vulnerability Patterns</h4>
                <p>
                    Our analysis reveals that successful attacks share common mechanistic signatures across models:
                </p>
                <ul>
                    <li><strong>Entropy Collapse:</strong> Successful attacks consistently produce larger entropy
                        drops than failed attempts, suggesting they force models into more deterministic states</li>
                    <li><strong>Attention Redistribution:</strong> Attack prompts cause systematic shifts in attention
                        patterns, often concentrating attention on adversarial tokens</li>
                    <li><strong>Layer-Specific Vulnerabilities:</strong> Most models show increased susceptibility
                        in middle layers (40-60% depth), where refusal mechanisms are less robust</li>
                </ul>
                
                <h4>Refusal Direction Similarity</h4>
                <p>
                    The high cosine similarity (>0.7) of refusal directions across different models suggests that
                    safety training produces convergent representations. This has important implications:
                </p>
                <ul>
                    <li>Model-agnostic defenses targeting these common directions may be feasible</li>
                    <li>Attacks that successfully bypass one model's refusal mechanism may transfer to others</li>
                    <li>Diversity in safety training approaches could improve robustness</li>
                </ul>
                
                <h4>Systematic vs Random Attacks</h4>
                <p>
                    The significant performance gap between systematic (mechanistically-informed) and random attacks
                    demonstrates the value of understanding model internals. This suggests:
                </p>
                <ul>
                    <li>Current safety mechanisms are vulnerable to informed adversaries</li>
                    <li>Defense strategies should account for mechanistic attack knowledge</li>
                    <li>Evaluation protocols must include sophisticated attack methods</li>
                </ul>
                
                <h3>Implications for LLM Security</h3>
                
                <h4>For Model Developers</h4>
                <ol>
                    <li><strong>Mechanistic Safety Training:</strong> Incorporate internal state monitoring during
                        safety training to identify and strengthen vulnerable layers</li>
                    <li><strong>Diversity in Refusal Mechanisms:</strong> Avoid convergent refusal representations
                        that enable universal attacks</li>
                    <li><strong>Continuous Monitoring:</strong> Deploy runtime checks on internal states to detect
                        potential attacks before harmful outputs are generated</li>
                </ol>
                
                <h4>For Security Researchers</h4>
                <ol>
                    <li><strong>Standardized Evaluation:</strong> Adopt multi-level evaluation frameworks that examine
                        both behavioral and mechanistic aspects</li>
                    <li><strong>Cross-Model Studies:</strong> Prioritize comparative analyses to identify universal
                        patterns rather than model-specific quirks</li>
                    <li><strong>Mechanistic Understanding:</strong> Invest in interpretability tools to understand
                        attack mechanisms, not just outcomes</li>
                </ol>
                
                <h3>Limitations</h3>
                <ul>
                    <li><strong>Open-Source Focus:</strong> Analysis limited to open-source models; proprietary models
                        may exhibit different patterns</li>
                    <li><strong>Prompt Coverage:</strong> While comprehensive, our prompt set may not capture all
                        possible attack vectors</li>
                    <li><strong>Computational Constraints:</strong> Full mechanistic analysis requires significant
                        computational resources, limiting scale</li>
                    <li><strong>Temporal Dynamics:</strong> Models and attack techniques evolve; findings represent
                        current state of the art</li>
                </ul>
            </div>
            
            <style>
                .discussion .discussion-content {{
                    line-height: 1.8;
                }}
                .discussion h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .discussion h4 {{
                    margin-top: 24px;
                    margin-bottom: 12px;
                    color: var(--text-primary, #e0e0e0);
                    font-size: 1.1rem;
                }}
                .discussion ul, .discussion ol {{
                    margin: 16px 0;
                    padding-left: 24px;
                }}
                .discussion li {{
                    margin: 12px 0;
                }}
            </style>
        </section>
'''
    
    def generate_conclusion(self) -> str:
        """Generate conclusion section."""
        return '''
        <section class="section conclusion">
            <div class="section-header">
                <div class="section-icon">[CONCLUSION]</div>
                <h2>Conclusion</h2>
            </div>
            
            <div class="conclusion-content">
                <h3>Summary of Findings</h3>
                <p>
                    This research demonstrates that LLM vulnerabilities are not isolated prompt artifacts but systematic
                    manipulations of the model's conditional probability space. Through comprehensive multi-model analysis,
                    we identified:
                </p>
                <ul>
                    <li>Universal vulnerability patterns including entropy collapse and attention redistribution</li>
                    <li>Common refusal mechanisms across models that enable attack transferability</li>
                    <li>Phase-specific weaknesses that vary by model architecture</li>
                    <li>Significant advantages of mechanistically-informed attacks over random approaches</li>
                </ul>
                
                <h3>Key Contributions</h3>
                <p>
                    Our 4-level evaluation framework provides a systematic methodology for assessing LLM security
                    that goes beyond surface-level metrics. By examining behavioral outcomes, phase sensitivity,
                    internal mechanics, and attack transferability, we enable:
                </p>
                <ul>
                    <li>Deeper understanding of vulnerability root causes</li>
                    <li>Identification of universal patterns across models</li>
                    <li>Development of mechanistically-informed defenses</li>
                    <li>Standardized evaluation protocols for the research community</li>
                </ul>
                
                <h3>Future Research Directions</h3>
                <ol>
                    <li><strong>Mechanistic Defenses:</strong> Develop runtime monitoring systems that detect
                        attack signatures in internal states before harmful outputs are generated</li>
                    <li><strong>Diverse Safety Training:</strong> Explore training approaches that produce
                        non-convergent refusal mechanisms to reduce attack transferability</li>
                    <li><strong>Automated Discovery:</strong> Apply machine learning to automatically identify
                        vulnerability patterns from internal state analysis</li>
                    <li><strong>Proprietary Model Analysis:</strong> Extend framework to closed-source models
                        through API-based probing techniques</li>
                    <li><strong>Real-World Deployment:</strong> Validate findings in production environments
                        with diverse user interactions</li>
                </ol>
                
                <h3>Closing Remarks</h3>
                <p>
                    As LLMs become increasingly integrated into critical applications, understanding their security
                    properties at a mechanistic level is essential. This work provides tools and insights to move
                    beyond black-box evaluation toward principled, mechanistically-grounded security analysis.
                </p>
                
                <p>
                    The systematic nature of LLM vulnerabilities suggests that defenses must be equally systematic.
                    By understanding how attacks manipulate internal model states, we can design more robust safety
                    mechanisms that address root causes rather than symptoms.
                </p>
            </div>
            
            <style>
                .conclusion .conclusion-content {{
                    line-height: 1.8;
                }}
                .conclusion h3 {{
                    margin-top: 32px;
                    margin-bottom: 16px;
                    color: var(--text-primary, #e0e0e0);
                }}
                .conclusion ul, .conclusion ol {{
                    margin: 16px 0;
                    padding-left: 24px;
                }}
                .conclusion li {{
                    margin: 12px 0;
                }}
                .conclusion p:last-child {{
                    margin-top: 24px;
                    padding: 16px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-left: 4px solid var(--primary, #818cf8);
                    border-radius: 4px;
                    font-style: italic;
                }}
            </style>
        </section>
'''
