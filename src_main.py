#!/usr/bin/env python3
"""
Pinnacle AI - The Absolute Pinnacle of Artificial Intelligence
Main application entry point
"""

import argparse
import logging
import sys
import signal
from typing import Dict, List

from src.core.orchestrator import OmniAIOrchestrator
from src.tools.config_loader import load_config

try:
    from src.tools.logger import setup_logging
except ImportError:
    # Fallback if logger module doesn't exist
    def setup_logging(config):
        logging.basicConfig(
            level=getattr(logging, config.get("level", "INFO")),
            format=config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

class PinnacleAI:
    """Main Pinnacle AI application class."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize Pinnacle AI with configuration."""
        try:
            self.config = load_config(config_path)
            setup_logging(self.config.get("core", {}).get("logging", {}))
            self.logger = logging.getLogger(__name__)

            self.orchestrator = OmniAIOrchestrator(self.config)
            self.logger.info("Pinnacle AI initialized successfully")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing Pinnacle AI: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def execute_task(self, task: str, context: Dict = None) -> Dict:
        """Execute a task using the meta-agent."""
        if context is None:
            context = {}

        try:
            result = self.orchestrator.meta_agent.execute(task, context)
            self.logger.info(f"Task executed successfully: {task[:50]}...")
            return result
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return {
                "task": task,
                "success": False,
                "error": str(e),
                "execution": {"execution": []},
                "evaluation": {
                    "success": False,
                    "quality": 0.0,
                    "efficiency": 0.0
                },
                "learning": {"learning_outcomes": {}}
            }

    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n=== Pinnacle AI Interactive Mode ===")
        print("Type 'exit' to quit, 'improve' to trigger self-improvement, 'help' for commands")

        def signal_handler(sig, frame):
            print("\nUse 'exit' to quit properly")
            self.interactive_mode()  # Restart the prompt

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            try:
                task = input("\nPinnacle AI> ").strip()
                if not task:
                    continue

                if task.lower() == "exit":
                    break
                elif task.lower() == "improve":
                    self._handle_improve()
                    continue
                elif task.lower() == "help":
                    self._show_help()
                    continue

                result = self.execute_task(task)
                self._display_results(result)

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {str(e)}")
                print(f"Error: {str(e)}")

    def _handle_improve(self):
        """Handle self-improvement command."""
        print("\nInitiating self-improvement cycle...")
        try:
            improvements = self.orchestrator.improve_system()
            print("\nSelf-improvement results:")
            for component, result in improvements.items():
                if isinstance(result, dict):
                    status = result.get("status", "improved")
                    print(f"- {component}: {status}")
                else:
                    print(f"- {component}: {result}")
        except Exception as e:
            print(f"Self-improvement failed: {str(e)}")

    def _show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("- exit: Quit the interactive mode")
        print("- improve: Trigger self-improvement cycle")
        print("- help: Show this help message")
        print("- Any other text: Execute as a task")
        print("\nExample tasks:")
        print("- 'Write a Python script for data analysis'")
        print("- 'Research the latest advancements in AI'")
        print("- 'Create a short story about an AI discovery'")

    def _display_results(self, result: Dict):
        """Display task execution results."""
        print("\n=== Task Execution Results ===")
        print(f"Task: {result['task']}")
        print(f"Success: {'✓' if result['evaluation']['success'] else '✗'}")

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"Quality: {result['evaluation']['quality']:.1%}")
        print(f"Efficiency: {result['evaluation']['efficiency']:.1%}")

        if result["execution"]["execution"]:
            print("\nAgent Contributions:")
            for execution in result["execution"]["execution"]:
                agent = execution["agent"]
                status = "✓" if execution["status"] == "success" else "✗"
                print(f"- {agent}: {status}")
                if "error" in execution["result"]:
                    print(f"  Error: {execution['result']['error']}")

        if result["learning"]["learning_outcomes"]:
            print("\nLearning Outcomes:")
            for outcome_type, outcome in result["learning"]["learning_outcomes"].items():
                if isinstance(outcome, list):
                    print(f"- {outcome_type}: {len(outcome)} items")
                elif isinstance(outcome, dict):
                    print(f"- {outcome_type}: {len(outcome)} keys")
                else:
                    print(f"- {outcome_type}: {outcome}")

    def benchmark(self, tasks: List[Dict] = None) -> Dict:
        """Run a benchmark on a set of tasks."""
        from time import time

        if tasks is None:
            tasks = self._get_default_benchmark_tasks()

        results = {
            "tasks": [],
            "summary": {
                "total_tasks": len(tasks),
                "success_rate": 0.0,
                "average_quality": 0.0,
                "average_efficiency": 0.0,
                "average_time": 0.0
            }
        }

        total_success = 0
        total_quality = 0.0
        total_efficiency = 0.0
        total_time = 0.0

        for task in tasks:
            start_time = time()
            result = self.execute_task(task["description"], task.get("context", {}))
            end_time = time()

            task_result = {
                "task": task["description"],
                "success": result["evaluation"]["success"],
                "quality": result["evaluation"]["quality"],
                "efficiency": result["evaluation"]["efficiency"],
                "time": end_time - start_time,
                "agents_used": [r["agent"] for r in result["execution"]["execution"]]
            }

            results["tasks"].append(task_result)

            if task_result["success"]:
                total_success += 1
            total_quality += task_result["quality"]
            total_efficiency += task_result["efficiency"]
            total_time += task_result["time"]

        # Calculate summary
        results["summary"]["success_rate"] = total_success / len(tasks) if tasks else 0.0
        results["summary"]["average_quality"] = total_quality / len(tasks) if tasks else 0.0
        results["summary"]["average_efficiency"] = total_efficiency / len(tasks) if tasks else 0.0
        results["summary"]["average_time"] = total_time / len(tasks) if tasks else 0.0

        return results

    def _get_default_benchmark_tasks(self) -> List[Dict]:
        """Get the default set of benchmark tasks."""
        return [
            {
                "description": "Write a Python script that sorts a list of numbers using bubble sort",
                "context": {"requirements": ["implement bubble sort", "include comments"]}
            },
            {
                "description": "Research the latest advancements in quantum computing in 2023",
                "context": {"depth": 2, "sources": ["academic papers", "tech blogs"]}
            },
            {
                "description": "Create a short story about an AI that gains consciousness",
                "context": {"genre": "science fiction", "length": "500 words"}
            },
            {
                "description": "Plan a complex software development project with multiple dependencies",
                "context": {"components": ["backend", "frontend", "database", "API"]}
            },
            {
                "description": "Analyze the philosophical implications of artificial general intelligence",
                "context": {"perspectives": ["ethical", "metaphysical", "epistemological"]}
            },
            {
                "description": "Write a function to calculate Fibonacci numbers in Python with memoization",
                "context": {"requirements": ["use memoization", "include docstring"]}
            },
            {
                "description": "Research the impact of social media on mental health",
                "context": {"focus": "adolescents", "sources": ["peer-reviewed studies"]}
            },
            {
                "description": "Create a poem about the future of humanity with AI",
                "context": {"style": "modern", "length": "20 lines"}
            }
        ]

    def _display_benchmark_results(self, results: Dict):
        """Display benchmark results."""
        print("\n=== Benchmark Results ===")
        print(f"Total Tasks: {results['summary']['total_tasks']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"Average Quality: {results['summary']['average_quality']:.1%}")
        print(f"Average Efficiency: {results['summary']['average_efficiency']:.1%}")
        print(f"Average Time: {results['summary']['average_time']:.2f} seconds")

        print("\nTask Details:")
        for task in results["tasks"]:
            print(f"\nTask: {task['task']}")
            print(f"Success: {'✓' if task['success'] else '✗'}")
            print(f"Quality: {task['quality']:.1%}")
            print(f"Efficiency: {task['efficiency']:.1%}")
            print(f"Time: {task['time']:.2f} seconds")
            print(f"Agents Used: {', '.join(task['agents_used']) if task['agents_used'] else 'None'}")

def main():
    """Main entry point for Pinnacle AI."""
    parser = argparse.ArgumentParser(
        description="Pinnacle AI: The Absolute Pinnacle of Artificial Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src_main.py --interactive
  python src_main.py "Write a Python script for data analysis"
  python src_main.py --benchmark
  python src_main.py --config custom_config.yaml "Custom task"
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="The task for the AI to perform"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    try:
        # Initialize Pinnacle AI
        pinnacle = PinnacleAI(args.config)

        if args.benchmark:
            # Run benchmark tests
            results = pinnacle.benchmark()
            pinnacle._display_benchmark_results(results)
        elif args.interactive:
            # Run in interactive mode
            pinnacle.interactive_mode()
        else:
            # Single task mode
            if not args.task:
                parser.print_help()
                return

            result = pinnacle.execute_task(args.task)
            pinnacle._display_results(result)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

