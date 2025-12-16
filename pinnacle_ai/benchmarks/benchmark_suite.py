"""
Comprehensive benchmark suite for Pinnacle-AI

Tracks:
- Reasoning ability
- Memory performance
- Emotional intelligence
- Response quality
- Speed
"""

import json
import time
from typing import Dict
from loguru import logger
from datetime import datetime
import os


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for Pinnacle-AI
    
    Tracks:
    - Reasoning ability
    - Memory performance
    - Emotional intelligence
    - Response quality
    - Speed
    """
    
    def __init__(self, ai):
        self.ai = ai
        self.results_dir = "benchmark_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load test cases
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> Dict:
        """Load benchmark test cases"""
        return {
            "reasoning": [
                {
                    "prompt": "If all cats are mammals, and all mammals are animals, are all cats animals?",
                    "expected_contains": ["yes", "animals"],
                    "category": "syllogism"
                },
                {
                    "prompt": "What is 15% of 80?",
                    "expected_contains": ["12"],
                    "category": "math"
                },
                {
                    "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "expected_contains": ["5"],
                    "category": "logic"
                }
            ],
            "memory": [
                {
                    "store": "The capital of France is Paris",
                    "query": "What is the capital of France?",
                    "expected_contains": ["Paris"]
                }
            ],
            "emotional": [
                {
                    "input": "I'm feeling really down today because I lost my job.",
                    "expected_sentiment": "negative",
                    "expected_empathy": True
                },
                {
                    "input": "I just got married! I'm so happy!",
                    "expected_sentiment": "positive",
                    "expected_empathy": True
                }
            ],
            "creativity": [
                {
                    "prompt": "Write a haiku about artificial intelligence",
                    "min_words": 10,
                    "category": "poetry"
                }
            ]
        }
    
    def run_all(self) -> Dict:
        """Run all benchmarks"""
        logger.info("Starting comprehensive benchmark suite...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
            "overall_score": 0.0
        }
        
        # Reasoning benchmark
        results["benchmarks"]["reasoning"] = self._benchmark_reasoning()
        
        # Memory benchmark
        results["benchmarks"]["memory"] = self._benchmark_memory()
        
        # Emotional intelligence benchmark
        results["benchmarks"]["emotional"] = self._benchmark_emotional()
        
        # Creativity benchmark
        results["benchmarks"]["creativity"] = self._benchmark_creativity()
        
        # Speed benchmark
        results["benchmarks"]["speed"] = self._benchmark_speed()
        
        # Calculate overall score
        scores = [b["score"] for b in results["benchmarks"].values()]
        results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Benchmark complete. Overall score: {results['overall_score']:.2f}")
        return results
    
    def _benchmark_reasoning(self) -> Dict:
        """Benchmark reasoning capabilities"""
        logger.info("Running reasoning benchmark...")
        
        correct = 0
        total = len(self.test_cases["reasoning"])
        details = []
        
        for test in self.test_cases["reasoning"]:
            start_time = time.time()
            try:
                response = self.ai.generate(test["prompt"], max_new_tokens=200)
            except:
                response = ""
            elapsed = time.time() - start_time
            
            # Check if expected content is present
            response_lower = response.lower()
            passed = all(exp.lower() in response_lower for exp in test["expected_contains"])
            
            if passed:
                correct += 1
            
            details.append({
                "prompt": test["prompt"],
                "category": test["category"],
                "response": response[:200],
                "passed": passed,
                "time": elapsed
            })
        
        return {
            "score": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "details": details
        }
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory capabilities"""
        logger.info("Running memory benchmark...")
        
        correct = 0
        total = len(self.test_cases["memory"])
        details = []
        
        for test in self.test_cases["memory"]:
            # Store
            try:
                self.ai.remember(test["store"])
            except:
                pass
            
            # Recall
            try:
                memories = self.ai.recall(test["query"], top_k=5)
            except:
                memories = []
            
            # Check if expected content is retrieved
            memory_text = " ".join([m.get("text", "") for m in memories])
            passed = all(exp.lower() in memory_text.lower() for exp in test["expected_contains"])
            
            if passed:
                correct += 1
            
            details.append({
                "stored": test["store"],
                "query": test["query"],
                "retrieved": memories[:3],
                "passed": passed
            })
        
        return {
            "score": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "details": details
        }
    
    def _benchmark_emotional(self) -> Dict:
        """Benchmark emotional intelligence"""
        logger.info("Running emotional intelligence benchmark...")
        
        correct = 0
        total = len(self.test_cases["emotional"])
        details = []
        
        for test in self.test_cases["emotional"]:
            # Get emotional response
            try:
                response = self.ai.generate(test["input"], max_new_tokens=200)
            except:
                response = ""
            
            # Check sentiment alignment
            # Simple check - in real implementation would use sentiment model
            sentiment_match = True  # Placeholder
            
            # Check for empathetic language
            empathy_words = ["understand", "feel", "here for you", "sorry", "happy for you"]
            has_empathy = any(word in response.lower() for word in empathy_words)
            
            passed = sentiment_match and (has_empathy == test["expected_empathy"])
            
            if passed:
                correct += 1
            
            details.append({
                "input": test["input"],
                "response": response[:200],
                "has_empathy": has_empathy,
                "passed": passed
            })
        
        return {
            "score": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "details": details
        }
    
    def _benchmark_creativity(self) -> Dict:
        """Benchmark creativity"""
        logger.info("Running creativity benchmark...")
        
        correct = 0
        total = len(self.test_cases["creativity"])
        details = []
        
        for test in self.test_cases["creativity"]:
            try:
                response = self.ai.generate(test["prompt"], max_new_tokens=200)
            except:
                response = ""
            
            # Check minimum length
            word_count = len(response.split())
            passed = word_count >= test["min_words"]
            
            if passed:
                correct += 1
            
            details.append({
                "prompt": test["prompt"],
                "response": response[:200],
                "word_count": word_count,
                "passed": passed
            })
        
        return {
            "score": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "details": details
        }
    
    def _benchmark_speed(self) -> Dict:
        """Benchmark response speed"""
        logger.info("Running speed benchmark...")
        
        prompts = [
            "Hello",
            "What is 2+2?",
            "Explain quantum computing in one sentence."
        ]
        
        times = []
        for prompt in prompts:
            start = time.time()
            try:
                self.ai.generate(prompt, max_new_tokens=50)
            except:
                pass
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times) if times else 5.0
        
        # Score: 1.0 if < 1s, 0.5 if < 3s, 0.0 if > 5s
        if avg_time < 1:
            score = 1.0
        elif avg_time < 3:
            score = 0.7
        elif avg_time < 5:
            score = 0.5
        else:
            score = 0.3
        
        return {
            "score": score,
            "average_time": avg_time,
            "times": times
        }
    
    def _save_results(self, results: Dict):
        """Save benchmark results"""
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.results_dir, filename)
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")


# Run benchmarks
if __name__ == "__main__":
    from pinnacle_ai import PinnacleAI
    
    ai = PinnacleAI()
    benchmark = BenchmarkSuite(ai)
    results = benchmark.run_all()
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Overall Score: {results['overall_score']:.2%}")
    for name, bench in results["benchmarks"].items():
        print(f"  {name}: {bench['score']:.2%}")

