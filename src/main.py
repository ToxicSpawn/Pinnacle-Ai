#!/usr/bin/env python3
"""
Enhanced main application for Pinnacle AI with all improvements
"""

import argparse
import logging
import sys
import signal
import time
from typing import Dict, List, Optional
from datetime import datetime

from src.core.orchestrator import OmniAIOrchestrator
from src.tools.config_loader import load_config
from src.security.security_manager import SecurityManager

try:
    from src.tools.logger import setup_logging
except ImportError:
    def setup_logging(config):
        logging.basicConfig(
            level=getattr(logging, config.get("level", "INFO")),
            format=config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )


class PinnacleAI:
    """Enhanced main Pinnacle AI application class"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize Pinnacle AI with comprehensive features"""
        try:
            self.config = load_config(config_path)
            setup_logging(self.config.get("core", {}).get("logging", {}))
            self.logger = logging.getLogger(__name__)
            self.security = SecurityManager(config_path)

            # Initialize orchestrator
            self.orchestrator = OmniAIOrchestrator(config_path)

            self.logger.info("Pinnacle AI initialized successfully")

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {config_path}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error initializing Pinnacle AI: {str(e)}")
            sys.exit(1)

    def execute_task(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute a task with enhanced security and monitoring"""
        if context is None:
            context = {}

        try:
            # Validate input
            if not self.security.validate_input(task, "text"):
                return self._create_error_result(task, "Invalid task input detected")

            # Add security context
            context["security"] = {
                "user_id": context.get("user_id", "anonymous"),
                "session_id": context.get("session_id", f"session_{int(time.time())}")
            }

            # Log task execution
            self._log_task_execution(task, context)

            # Execute task
            result = self.orchestrator.execute_task(task, context)

            # Log completion
            self._log_task_completion(result)

            return result

        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            self._log_task_failure(task, str(e))
            return self._create_error_result(task, str(e))

    def _create_error_result(self, task: str, error: str) -> Dict:
        """Create a standardized error result"""
        return {
            "task": task,
            "status": "error",
            "error": error,
            "execution": {"execution": []},
            "evaluation": {
                "success": False,
                "quality": 0.0,
                "efficiency": 0.0
            },
            "learning": {"learning_outcomes": {}}
        }

    def _log_task_execution(self, task: str, context: Dict):
        """Log task execution for audit and monitoring"""
        self.logger.info(f"Executing task: {task[:50]}...")
        self.security.log_audit_event(
            "task_started",
            context.get("user_id", "anonymous"),
            {
                "task": task,
                "context": {k: v for k, v in context.items() if k != "security"},
                "timestamp": datetime.now().isoformat()
            }
        )

    def _log_task_completion(self, result: Dict):
        """Log task completion"""
        self.logger.info(f"Task completed: {result['task'][:50]}... (Success: {result['evaluation']['success']})")
        self.security.log_audit_event(
            "task_completed",
            result.get("execution", {}).get("execution", [{}])[0].get("user_id", "anonymous") if result.get("execution", {}).get("execution") else "anonymous",
            {
                "task": result["task"],
                "success": result["evaluation"]["success"],
                "quality": result["evaluation"]["quality"],
                "efficiency": result["evaluation"]["efficiency"],
                "timestamp": datetime.now().isoformat()
            }
        )

    def _log_task_failure(self, task: str, error: str):
        """Log task failure"""
        self.logger.error(f"Task failed: {task[:50]}... (Error: {error})")
        self.security.log_audit_event(
            "task_failed",
            "anonymous",
            {
                "task": task,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
        )

    def benchmark(self, tasks: Optional[List[Dict]] = None) -> Dict:
        """Run comprehensive benchmark tests"""
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
                "average_time": 0.0,
                "system_metrics": []
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
        """Get the default set of comprehensive benchmark tasks"""
        return [
            {
                "description": "Write a Python script that sorts a list of numbers using bubble sort and includes unit tests",
                "context": {
                    "requirements": ["implement bubble sort", "include unit tests", "add docstrings"],
                    "language": "Python"
                }
            },
            {
                "description": "Research the latest advancements in quantum computing in 2023-2024, focusing on error correction and practical applications",
                "context": {
                    "depth": 3,
                    "sources": ["academic papers", "tech blogs", "patents"],
                    "time_period": "2023-2024"
                }
            },
            {
                "description": "Create a comprehensive business plan for an AI-powered personalized education startup targeting K-12 students",
                "context": {
                    "sections": [
                        "executive summary", "market analysis", "product description",
                        "business model", "marketing strategy", "operations plan",
                        "financial projections", "risk analysis"
                    ],
                    "length": "20-30 pages",
                    "format": "investor-ready"
                }
            },
            {
                "description": "Design a scientific study to investigate the effects of social media usage on adolescent mental health, including methodology and ethical considerations",
                "context": {
                    "focus": "adolescents aged 13-18",
                    "duration": "12 months",
                    "methods": ["survey", "longitudinal study", "neuroimaging"]
                }
            },
            {
                "description": "Write a 1000-word short story about an AI that gains consciousness and explores human emotions, in the style of Isaac Asimov",
                "context": {
                    "genre": "science fiction",
                    "style": "Isaac Asimov",
                    "themes": ["consciousness", "human-AI interaction", "ethics"],
                    "length": "1000 words"
                }
            },
            {
                "description": "Create a complete technical architecture for a distributed AI system that can handle real-time multi-modal processing",
                "context": {
                    "requirements": [
                        "scalability", "low latency", "fault tolerance",
                        "multi-modal processing", "real-time capabilities"
                    ],
                    "components": [
                        "ingestion layer", "processing layer",
                        "storage layer", "serving layer", "monitoring"
                    ]
                }
            },
            {
                "description": "Analyze the philosophical implications of artificial general intelligence from ethical, metaphysical, and epistemological perspectives",
                "context": {
                    "perspectives": ["ethical", "metaphysical", "epistemological"],
                    "philosophers": ["Searle", "Dennett", "Bostrom", "Chalmers"],
                    "format": "academic essay"
                }
            },
            {
                "description": "Develop a comprehensive marketing strategy for a new AI-powered healthcare diagnostic tool, including target audience, channels, and messaging",
                "context": {
                    "product": "AI-powered healthcare diagnostic tool",
                    "target_audience": ["hospitals", "clinics", "insurance companies"],
                    "budget": "$5M",
                    "timeframe": "12 months"
                }
            }
        ]

    def interactive_mode(self):
        """Run in enhanced interactive mode with comprehensive features"""
        print("\n=== Pinnacle AI Interactive Mode ===")
        print("Type 'exit' to quit, 'improve' to trigger self-improvement, 'help' for commands")
        print("Type 'template' to use task templates, 'benchmark' to run benchmarks")

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
                elif task.lower() == "benchmark":
                    self._run_interactive_benchmark()
                    continue
                elif task.lower() == "status":
                    self._show_system_status()
                    continue
                elif task.lower() == "agents":
                    self._show_agents()
                    continue

                result = self.execute_task(task)
                self._display_results(result)

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {str(e)}")
                print(f"Error: {str(e)}")

    def _handle_improve(self):
        """Handle self-improvement command with enhanced safety"""
        print("\nInitiating self-improvement cycle...")
        try:
            # Check if improvement is allowed
            if not self.config.get("self_evolution", {}).get("active", True):
                print("Self-improvement is currently disabled in configuration")
                return

            # Confirm with user
            confirm = input("This may modify system components. Continue? (y/n): ").lower()
            if confirm != 'y':
                print("Self-improvement cancelled")
                return

            # Run improvement
            improvements = self.orchestrator.improve_system()

            print("\nSelf-improvement results:")
            for component, result in improvements.items():
                if isinstance(result, dict):
                    status = result.get("status", "improved")
                    print(f"- {component}: {status}")
                    if "details" in result:
                        print(f"  Details: {result['details']}")
                else:
                    print(f"- {component}: {result}")

        except Exception as e:
            print(f"Self-improvement failed: {str(e)}")

    def _show_help(self):
        """Show enhanced help information"""
        print("\nAvailable commands:")
        print("- exit: Quit the interactive mode")
        print("- improve: Trigger self-improvement cycle")
        print("- benchmark: Run benchmark tests")
        print("- status: Show system status")
        print("- agents: Show available agents")
        print("- help: Show this help message")
        print("- Any other text: Execute as a task")

        print("\nExample tasks:")
        print("- 'Write a Python web application using FastAPI'")
        print("- 'Research the latest advancements in neurosymbolic AI'")
        print("- 'Create a business plan for an AI startup'")
        print("- 'Design a scientific experiment to test a new hypothesis'")
        print("- 'Compose a short story about artificial consciousness'")

    def _run_interactive_benchmark(self):
        """Run benchmark in interactive mode"""
        print("\nRunning benchmark tests...")
        try:
            results = self.benchmark()

            print("\n=== Benchmark Results ===")
            print(f"Total Tasks: {results['summary']['total_tasks']}")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            print(f"Average Quality: {results['summary']['average_quality']:.1%}")
            print(f"Average Efficiency: {results['summary']['average_efficiency']:.1%}")
            print(f"Average Time: {results['summary']['average_time']:.2f} seconds")

            print("\nTask Details:")
            for task in results["tasks"]:
                status = "✓" if task["success"] else "✗"
                print(f"\nTask: {task['task'][:50]}...")
                print(f"Status: {status}")
                print(f"Quality: {task['quality']:.1%}")
                print(f"Efficiency: {task['efficiency']:.1%}")
                print(f"Time: {task['time']:.2f} seconds")
                print(f"Agents Used: {', '.join(task['agents_used']) if task['agents_used'] else 'None'}")

        except Exception as e:
            print(f"Benchmark failed: {str(e)}")

    def _show_system_status(self):
        """Show comprehensive system status"""
        try:
            status = self.orchestrator.get_system_status()

            print("\n=== System Status ===")
            print(f"Status: {status.get('status', 'operational').upper()}")

            print("\nCore Components:")
            for component, comp_status in status.get("components", {}).items():
                if isinstance(comp_status, dict):
                    print(f"- {component}: {comp_status.get('status', 'unknown')}")
                else:
                    print(f"- {component}: operational")

            print("\nAgents:")
            for agent, agent_status in status.get("agents", {}).items():
                if isinstance(agent_status, dict):
                    perf = agent_status.get("performance", {})
                    success_rate = perf.get("success_rate", 0)
                    print(f"- {agent}: Success Rate {success_rate:.1%}")
                else:
                    print(f"- {agent}: operational")

            print("\nPerformance Metrics:")
            perf = status.get("performance", {})
            if isinstance(perf, dict):
                print(f"- Status: {perf.get('status', 'optimal')}")

            print("\nSecurity Status:")
            sec = status.get("security", {})
            print(f"- Status: {sec.get('status', 'operational')}")

        except Exception as e:
            print(f"Failed to get system status: {str(e)}")

    def _show_agents(self):
        """Show information about available agents"""
        try:
            agents = self.orchestrator.agents

            print("\n=== Available Agents ===")
            for name, agent in agents.items():
                if hasattr(agent, "get_status"):
                    status = agent.get_status()
                    perf = status.get("performance", {}) if isinstance(status, dict) else {}
                    print(f"\nAgent: {name}")
                    print(f"Status: {status.get('status', 'unknown') if isinstance(status, dict) else 'operational'}")
                    if perf:
                        print(f"Success Rate: {perf.get('success_rate', 0):.1%}")
                        print(f"Quality: {perf.get('avg_quality', 0):.1%}")
                        print(f"Efficiency: {perf.get('avg_efficiency', 0):.1%}")
                else:
                    print(f"\nAgent: {name}")
                    print(f"Status: operational")

        except Exception as e:
            print(f"Failed to get agent information: {str(e)}")

    def _display_results(self, result: Dict):
        """Display enhanced task execution results"""
        print("\n=== Task Execution Results ===")
        print(f"Task: {result['task']}")
        print(f"Status: {'✓' if result.get('evaluation', {}).get('success', False) else '✗'}")

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"Quality: {result.get('evaluation', {}).get('quality', 0):.1%}")
        print(f"Efficiency: {result.get('evaluation', {}).get('efficiency', 0):.1%}")

        if result.get("execution", {}).get("execution"):
            print("\nAgent Contributions:")
            for execution in result["execution"]["execution"]:
                agent = execution["agent"]
                status = "✓" if execution["status"] == "success" else "✗"
                time_taken = execution.get("execution_time", 0)
                print(f"- {agent}: {status} ({time_taken:.2f}s)")
                if "error" in execution.get("result", {}):
                    print(f"  Error: {execution['result']['error']}")
                if "optimizations" in execution:
                    print(f"  Optimizations: {', '.join(execution['optimizations'])}")

        if result.get("learning", {}).get("learning_outcomes"):
            print("\nLearning Outcomes:")
            for outcome_type, outcome in result["learning"]["learning_outcomes"].items():
                if isinstance(outcome, list):
                    print(f"- {outcome_type}: {len(outcome)} items")
                elif isinstance(outcome, dict):
                    print(f"- {outcome_type}: {len(outcome)} keys")
                else:
                    print(f"- {outcome_type}: {outcome}")

        if "performance" in result:
            print("\nPerformance:")
            perf = result["performance"]
            print(f"- Execution Time: {perf.get('execution_time', 0):.2f}s")
            if "optimizations_applied" in perf:
                print(f"- Optimizations Applied: {len(perf['optimizations_applied'])}")

    def launch_web_ui(self):
        """Launch the enhanced web interface"""
        try:
            from web_ui_enhanced import EnhancedWebUI
            ui = EnhancedWebUI()
            ui.launch()
        except ImportError:
            try:
                from web_ui import launch_web_ui
                launch_web_ui()
            except ImportError:
                self.logger.error("Web UI not available. Install gradio: pip install gradio")
                print("Error: Web UI not available. Install gradio: pip install gradio")
        except Exception as e:
            self.logger.error(f"Failed to launch web UI: {str(e)}")
            print(f"Error launching web UI: {str(e)}")

    def run_api_server(self):
        """Run the FastAPI server"""
        try:
            import uvicorn
            from web.api.main import app
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                workers=self.config.get("deployment", {}).get("workers", 4)
            )
        except ImportError:
            self.logger.error("API server not available. Install fastapi and uvicorn")
            print("Error: API server not available. Install fastapi and uvicorn")
        except Exception as e:
            self.logger.error(f"Failed to run API server: {str(e)}")
            print(f"Error running API server: {str(e)}")


def main():
    """Main entry point for Pinnacle AI"""
    parser = argparse.ArgumentParser(
        description="Pinnacle AI: The Absolute Pinnacle of Artificial Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --interactive
  python src/main.py "Write a Python script for data analysis"
  python src/main.py --benchmark
  python src/main.py --web
  python src/main.py --api
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
        "--web",
        action="store_true",
        help="Launch web interface"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API server"
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

        if args.web:
            # Launch web interface
            pinnacle.launch_web_ui()
        elif args.api:
            # Run API server
            pinnacle.run_api_server()
        elif args.benchmark:
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

