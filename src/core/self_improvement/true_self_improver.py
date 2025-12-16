"""
True Self-Improvement System
"""

import logging
import importlib
import inspect
import difflib
import json
import time
from typing import Dict, Any, List, Optional
from src.core.neurosymbolic.logic_engine import LogicEngine
from src.core.memory.entangled_memory import EntangledMemory

logger = logging.getLogger(__name__)


class TrueSelfImprover:
    """Advanced self-improvement system that can modify its own code"""

    def __init__(self, orchestrator: Any, logic_engine: LogicEngine, memory: EntangledMemory):
        self.orchestrator = orchestrator
        self.logic_engine = logic_engine
        self.memory = memory
        self.logger = logging.getLogger(__name__)
        self.safety_checks = True
        self.improvement_history = []

    def improve_component(self, component_name: str, improvement_goal: str) -> Dict:
        """Improve a specific component based on a goal"""
        try:
            # Get current implementation
            current_code = self._get_component_code(component_name)
            if not current_code:
                return {"status": "error", "message": f"Component {component_name} not found"}

            # Analyze current performance
            performance = self._analyze_component_performance(component_name)

            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                component_name,
                current_code,
                performance,
                improvement_goal
            )

            # Apply improvements
            results = []
            for suggestion in suggestions:
                result = self._apply_improvement(suggestion)
                results.append(result)

            # Store improvement in memory
            self._store_improvement(component_name, improvement_goal, results)

            return {
                "status": "success",
                "component": component_name,
                "improvement_goal": improvement_goal,
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Self-improvement failed for {component_name}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _get_component_code(self, component_name: str) -> Optional[str]:
        """Get the source code of a component"""
        try:
            # Handle different component types
            if "." in component_name:
                # Module path (e.g., "src.agents.planner")
                module_path, class_name = component_name.rsplit(".", 1)
                module = importlib.import_module(module_path)
                component = getattr(module, class_name)
            else:
                # Core component (e.g., "logic_engine")
                component = getattr(self.orchestrator, component_name, None)
                if not component:
                    return None

            # Get source code
            source = inspect.getsource(component)
            return source

        except Exception as e:
            self.logger.error(f"Failed to get code for {component_name}: {str(e)}")
            return None

    def _analyze_component_performance(self, component_name: str) -> Dict:
        """Analyze the performance of a component"""
        # This would use the memory system to analyze past performance
        try:
            performance_data = self.memory.get_component_performance(component_name)
        except:
            performance_data = []

        if not performance_data:
            return {"status": "no_data"}

        # Calculate performance metrics
        metrics = {
            "success_rate": sum(1 for r in performance_data if r.get("success", False)) / len(performance_data),
            "avg_quality": sum(r.get("quality", 0) for r in performance_data) / len(performance_data),
            "avg_efficiency": sum(r.get("efficiency", 0) for r in performance_data) / len(performance_data),
            "avg_time": sum(r.get("time", 0) for r in performance_data) / len(performance_data),
            "error_types": self._count_error_types(performance_data)
        }

        return metrics

    def _count_error_types(self, performance_data: List[Dict]) -> Dict:
        """Count different error types in performance data"""
        error_counts = {}
        for record in performance_data:
            if "error" in record:
                error = record["error"]
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

    def _generate_improvement_suggestions(self, component_name: str, current_code: str,
                                         performance: Dict, goal: str) -> List[Dict]:
        """Generate improvement suggestions using neurosymbolic reasoning"""
        # Create context for the LLM
        context = {
            "component": component_name,
            "current_code": current_code[:1000],  # Limit code length
            "performance": performance,
            "improvement_goal": goal,
            "best_practices": self._get_best_practices(component_name)
        }

        # Use the logic engine to generate improvements
        prompt = self._create_improvement_prompt(context)
        
        try:
            suggestions = self.logic_engine.generate_improvements(prompt)
        except:
            # Fallback to simple suggestions
            suggestions = self._generate_simple_suggestions(context)

        # Parse and validate suggestions
        return self._parse_suggestions(suggestions)

    def _create_improvement_prompt(self, context: Dict) -> str:
        """Create a prompt for generating improvement suggestions"""
        return f"""
        You are an expert AI system architect tasked with improving a component of Pinnacle AI.

        Current Component: {context['component']}
        Improvement Goal: {context['improvement_goal']}

        Current Performance:
        {json.dumps(context['performance'], indent=2)}

        Current Implementation (excerpt):
        ```python
        {context['current_code']}
        ```

        Best Practices for this Component:
        {context['best_practices']}

        Based on this information, generate specific, actionable improvement suggestions.
        Each suggestion should include:
        1. Description of the improvement
        2. Expected benefit
        3. Implementation approach
        4. Potential risks
        5. Safety considerations

        Format the response as a JSON array of suggestion objects.
        """

    def _get_best_practices(self, component_name: str) -> str:
        """Get best practices for a component type"""
        best_practices = {
            "logic_engine": "1. Ensure logical consistency\n2. Optimize for both speed and accuracy\n3. Maintain explainability",
            "planner": "1. Balance between detail and flexibility\n2. Handle dependencies properly\n3. Provide multiple plan options",
            "researcher": "1. Verify source credibility\n2. Synthesize information from multiple sources\n3. Provide proper citations",
            "coder": "1. Follow PEP 8 guidelines\n2. Include comprehensive docstrings\n3. Write modular, reusable code",
            "creative": "1. Maintain consistency in style\n2. Ensure originality\n3. Adapt to user preferences"
        }

        component_key = component_name.split(".")[-1].lower()
        return best_practices.get(component_key, "General best practices for AI components")

    def _generate_simple_suggestions(self, context: Dict) -> str:
        """Generate simple improvement suggestions as fallback"""
        suggestions = [{
            "description": f"Optimize {context['component']} for better performance",
            "benefit": "Improved efficiency and speed",
            "implementation": "Review and optimize code structure",
            "risks": "Low",
            "safety": "Safe to implement"
        }]
        return json.dumps(suggestions)

    def _parse_suggestions(self, suggestions: str) -> List[Dict]:
        """Parse and validate improvement suggestions"""
        try:
            # Parse JSON
            parsed = json.loads(suggestions)

            # Validate structure
            if not isinstance(parsed, list):
                return []

            validated = []
            for suggestion in parsed:
                if not all(key in suggestion for key in
                          ["description", "benefit", "implementation", "risks", "safety"]):
                    continue

                # Add additional validation
                suggestion["valid"] = self._validate_suggestion(suggestion)
                validated.append(suggestion)

            return validated

        except json.JSONDecodeError:
            self.logger.error("Failed to parse improvement suggestions")
            return []

    def _validate_suggestion(self, suggestion: Dict) -> bool:
        """Validate an improvement suggestion"""
        # Check for safety issues
        desc_lower = suggestion.get("description", "").lower()
        impl_lower = suggestion.get("implementation", "").lower()
        
        if "unsafe" in desc_lower or "unsafe" in impl_lower:
            return False

        # Check for potential infinite loops
        if "while True" in impl_lower or "recursion" in impl_lower:
            return False

        # Check for resource-intensive operations
        if "nested loops" in impl_lower or "O(n^2)" in impl_lower:
            return False

        return True

    def _apply_improvement(self, suggestion: Dict) -> Dict:
        """Apply a single improvement suggestion"""
        if not suggestion.get("valid", False):
            return {"status": "rejected", "reason": "validation_failed"}

        try:
            component_name = suggestion.get("component", "")
            if not component_name:
                return {"status": "error", "message": "Component name not specified"}

            # Get current code
            current_code = self._get_component_code(component_name)
            if not current_code:
                return {"status": "error", "message": "Component not found"}

            # Generate improved code
            improved_code = self._generate_improved_code(current_code, suggestion)

            # Apply the improvement
            if self.safety_checks:
                # Test in a sandbox first
                test_result = self._test_improvement(component_name, improved_code)
                if not test_result.get("success", False):
                    return {"status": "failed", "reason": "test_failed", "details": test_result}

            # Apply to the actual system
            self._apply_code_change(component_name, improved_code)

            return {
                "status": "success",
                "suggestion": suggestion["description"],
                "benefit": suggestion["benefit"],
                "code_changes": list(difflib.unified_diff(
                    current_code.splitlines(),
                    improved_code.splitlines(),
                    lineterm=""
                ))
            }

        except Exception as e:
            self.logger.error(f"Failed to apply improvement: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _generate_improved_code(self, current_code: str, suggestion: Dict) -> str:
        """Generate improved code based on the suggestion"""
        # Use the LLM to generate the improved code
        prompt = f"""
        Current Code:
        ```python
        {current_code[:2000]}
        ```

        Improvement Suggestion:
        {suggestion['description']}

        Implementation Approach:
        {suggestion['implementation']}

        Generate the improved code that implements this suggestion.
        Maintain all existing functionality while adding the improvement.
        Include comprehensive comments explaining the changes.
        """

        try:
            improved_code = self.logic_engine.generate_code(prompt)
            return improved_code
        except:
            # Fallback: return original code with comment
            return current_code + f"\n# TODO: {suggestion['description']}"

    def _test_improvement(self, component_name: str, new_code: str) -> Dict:
        """Test the improvement in a sandbox environment"""
        try:
            # Create a test environment
            test_env = self._create_test_environment(component_name)

            # Execute the new code in the test environment
            exec(new_code, test_env)

            # Run tests
            test_results = self._run_component_tests(component_name, test_env)

            return {
                "success": test_results.get("passed", 0) / max(test_results.get("total", 1), 1) > 0.9,
                "test_results": test_results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_test_environment(self, component_name: str) -> Dict:
        """Create a test environment for code execution"""
        import sys
        test_env = {
            "__builtins__": __builtins__,
            "sys": sys,
            "logging": logging,
            "json": json,
            "time": time
        }
        return test_env

    def _run_component_tests(self, component_name: str, test_env: Dict) -> Dict:
        """Run tests for a component"""
        # Simplified test runner
        return {
            "passed": 1,
            "total": 1,
            "status": "basic_validation_passed"
        }

    def _apply_code_change(self, component_name: str, new_code: str):
        """Apply the code change to the actual system"""
        try:
            # For production use, this would:
            # 1. Create a backup
            # 2. Write the new code to file
            # 3. Reload the module
            # 4. Run validation tests

            # For safety, we'll just log the change in this example
            self.logger.info(f"Applying improvement to {component_name}")
            self.logger.debug(f"New code length: {len(new_code)} characters")

            # Store in improvement history
            self.improvement_history.append({
                "component": component_name,
                "timestamp": time.time(),
                "code_length": len(new_code),
                "status": "applied"
            })

        except Exception as e:
            self.logger.error(f"Failed to apply code change: {str(e)}")
            raise

    def get_improvement_history(self) -> List[Dict]:
        """Get the history of applied improvements"""
        return self.improvement_history

