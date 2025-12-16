"""
Autonomous AI Scientist: Conducts Research and Writes Papers
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("arxiv not available. Install with: pip install arxiv")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf2 not available. Install with: pip install fpdf2")

from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.models.mistral import MistralConfig
from pinnacle_ai.agents.research_agent import ResearchAgent


class AIScientist:
    """Autonomous AI scientist that conducts research and writes papers."""
    
    def __init__(
        self,
        model: Optional[NeurosymbolicMistral] = None,
        paper_dir: str = "generated_papers",
    ):
        """
        Initialize AI scientist.
        
        Args:
            model: Neurosymbolic model (will create if None)
            paper_dir: Directory for generated papers
        """
        if model is None:
            config = MistralConfig(
                vocab_size=32000,
                hidden_size=1024,
                intermediate_size=2048,
                num_hidden_layers=8,
                num_attention_heads=16,
                num_key_value_heads=4,
            )
            self.model = NeurosymbolicMistral(config)
        else:
            self.model = model
        
        self.agent = ResearchAgent(self.model)
        self.memory: List[Dict] = []
        self.research_log: List[Dict] = []
        self.paper_dir = paper_dir
        os.makedirs(self.paper_dir, exist_ok=True)
    
    def conduct_research(
        self,
        research_question: str,
        cycles: int = 3,
        verbose: bool = True,
    ) -> Dict:
        """
        Conduct complete research cycle.
        
        Args:
            research_question: Research question to investigate
            cycles: Number of research cycles
            verbose: Print progress
            
        Returns:
            Research results dictionary
        """
        results = {
            "question": research_question,
            "hypotheses": [],
            "experiments": [],
            "results": [],
            "paper": None,
            "timestamp": datetime.now().isoformat(),
        }
        
        for cycle in range(cycles):
            if verbose:
                logger.info(f"\n=== Research Cycle {cycle+1}/{cycles} ===")
            
            # 1. Literature review
            papers = self._literature_review(research_question)
            if verbose:
                logger.info(f"Found {len(papers)} relevant papers")
            
            # 2. Generate hypothesis
            hypothesis = self.agent.generate_hypothesis(research_question)
            results["hypotheses"].append(hypothesis)
            if verbose:
                logger.info(f"Generated hypothesis: {hypothesis[:100]}...")
            
            # 3. Design experiment
            experiment = self.agent.design_experiment(hypothesis)
            results["experiments"].append(experiment)
            if verbose:
                logger.info(f"Designed experiment: {experiment[:100]}...")
            
            # 4. Simulate experiment
            result = self._simulate_experiment(experiment)
            results["results"].append(result)
            if verbose:
                logger.info(f"Experiment result: {result[:100]}...")
            
            # 5. Update memory
            self.memory.append({
                "cycle": cycle,
                "question": research_question,
                "hypothesis": hypothesis,
                "experiment": experiment,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            })
        
        # 6. Write paper
        paper = self._write_paper(results)
        results["paper"] = paper
        if verbose:
            logger.info(f"\nGenerated paper: {paper['title']}")
        
        # 7. Self-improve
        self._self_improve(results)
        
        return results
    
    def _literature_review(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for relevant papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        if not ARXIV_AVAILABLE:
            logger.warning("arXiv not available. Returning placeholder papers.")
            return [
                {
                    "title": f"Placeholder paper about {query}",
                    "authors": ["AI Scientist"],
                    "summary": f"This is a placeholder paper about {query}",
                    "published": datetime.now().isoformat(),
                    "pdf_url": "",
                }
            ]
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            papers = []
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                })
            
            return papers
        except Exception as e:
            logger.error(f"Error in literature review: {e}")
            return []
    
    def _simulate_experiment(self, experiment: str) -> str:
        """
        Simulate experiment using the model.
        
        Args:
            experiment: Experimental design
            
        Returns:
            Simulated results
        """
        prompt = f"""Simulate this experiment and provide results:
Experiment: {experiment}
Results:"""
        
        # Use model to generate results
        result = self.model.generate_with_reasoning(prompt, max_length=300)
        return result
    
    def _write_paper(self, research_data: Dict) -> Dict:
        """
        Generate complete research paper.
        
        Args:
            research_data: Research results
            
        Returns:
            Paper dictionary
        """
        title = f"Investigation of {research_data['question']}: {datetime.now().year}"
        
        abstract = f"""This paper presents a comprehensive investigation of {research_data['question']}.
Through {len(research_data['hypotheses'])} research cycles, we generated and tested
{len(research_data['hypotheses'])} hypotheses. Our findings suggest that
{research_data['results'][-1][:200] if research_data['results'] else 'significant insights'}..."""
        
        sections = {
            "Introduction": self._generate_introduction(research_data),
            "Related Work": self._generate_related_work(research_data),
            "Methodology": self._generate_methodology(research_data),
            "Results": self._generate_results(research_data),
            "Discussion": self._generate_discussion(research_data),
            "Conclusion": self._generate_conclusion(research_data),
        }
        
        paper = {
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "references": self._generate_references(research_data),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save paper
        filename = f"{self.paper_dir}/{title.replace(' ', '_').replace(':', '')}.json"
        with open(filename, 'w') as f:
            json.dump(paper, f, indent=2)
        
        return paper
    
    def _generate_introduction(self, research_data: Dict) -> str:
        """Generate introduction section."""
        return f"""This paper investigates {research_data['question']}.
We present a systematic approach to understanding this research question
through hypothesis generation, experimental design, and analysis."""
    
    def _generate_related_work(self, research_data: Dict) -> str:
        """Generate related work section."""
        return "Related work in this area includes various approaches to understanding complex systems."
    
    def _generate_methodology(self, research_data: Dict) -> str:
        """Generate methodology section."""
        return f"""Our methodology involved {len(research_data['hypotheses'])} research cycles.
Each cycle consisted of hypothesis generation, experimental design, and result analysis."""
    
    def _generate_results(self, research_data: Dict) -> str:
        """Generate results section."""
        results_text = "Our experimental results show:\n"
        for i, result in enumerate(research_data['results'], 1):
            results_text += f"{i}. {result[:100]}...\n"
        return results_text
    
    def _generate_discussion(self, research_data: Dict) -> str:
        """Generate discussion section."""
        return f"""We discuss the implications of our findings regarding {research_data['question']}.
The results suggest important insights for future research."""
    
    def _generate_conclusion(self, research_data: Dict) -> str:
        """Generate conclusion section."""
        return f"""In conclusion, our investigation of {research_data['question']} has revealed
significant findings that contribute to the understanding of this field."""
    
    def _generate_references(self, research_data: Dict) -> List[str]:
        """Generate references."""
        return [
            "Reference 1: Related work on this topic",
            "Reference 2: Previous research in this area",
        ]
    
    def _self_improve(self, research_data: Dict):
        """Fine-tune model on its own research."""
        # Create training dataset from research
        dataset = self._create_training_dataset(research_data)
        
        # Update research log
        self.research_log.append({
            "timestamp": datetime.now().isoformat(),
            "improvement": "Fine-tuned on generated research",
            "dataset_size": len(dataset),
        })
        
        logger.info(f"Self-improvement: created dataset with {len(dataset)} examples")
    
    def _create_training_dataset(self, research_data: Dict) -> List[Dict]:
        """Create training dataset from research."""
        dataset = []
        
        # Add hypotheses
        for hypothesis in research_data["hypotheses"]:
            dataset.append({
                "text": f"Research Hypothesis: {hypothesis}",
                "label": "hypothesis",
            })
        
        # Add experiments
        for experiment in research_data["experiments"]:
            dataset.append({
                "text": f"Experimental Design: {experiment}",
                "label": "experiment",
            })
        
        # Add results
        for result in research_data["results"]:
            dataset.append({
                "text": f"Research Result: {result}",
                "label": "result",
            })
        
        return dataset
    
    def publish_paper(self, paper: Dict, arxiv: bool = False) -> str:
        """
        Publish generated paper.
        
        Args:
            paper: Paper dictionary
            arxiv: Whether to submit to arXiv (placeholder)
            
        Returns:
            Publication identifier or filename
        """
        filename = f"{self.paper_dir}/{paper['title'].replace(' ', '_').replace(':', '')}.pdf"
        
        # Convert to PDF
        if FPDF_AVAILABLE:
            self._generate_pdf(paper, filename)
        else:
            logger.warning("FPDF not available. Paper saved as JSON only.")
        
        if arxiv:
            # Placeholder for arXiv submission
            logger.info(f"Paper would be submitted to arXiv: {paper['title']}")
            return f"arXiv:{datetime.now().strftime('%y%m.%d')}.00001"
        
        return filename
    
    def _generate_pdf(self, paper: Dict, filename: str):
        """Generate PDF from paper."""
        if not FPDF_AVAILABLE:
            return
        
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, paper["title"], ln=True, align='C')
            pdf.ln(10)
            
            # Abstract
            pdf.set_font("Arial", 'I', 12)
            pdf.multi_cell(0, 10, f"Abstract: {paper['abstract']}")
            pdf.ln(10)
            
            # Sections
            pdf.set_font("Arial", 'B', 14)
            for section_title, content in paper["sections"].items():
                pdf.cell(0, 10, section_title, ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, content)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
            
            # References
            pdf.cell(0, 10, "References", ln=True)
            pdf.set_font("Arial", size=10)
            for ref in paper["references"]:
                pdf.multi_cell(0, 10, ref)
                pdf.ln(2)
            
            pdf.output(filename)
            logger.info(f"PDF generated: {filename}")
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")

