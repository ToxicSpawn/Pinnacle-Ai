"""
Enhanced Web UI for Pinnacle AI
"""

import gradio as gr
from src_main import PinnacleAI
import json
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import io
    import base64
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EnhancedWebUI:
    """Next-generation web interface for Pinnacle AI"""

    def __init__(self):
        self.pinnacle = PinnacleAI()
        self.task_history = []
        self.current_task = None
        self.feedback_data = []

    def launch(self):
        """Launch the enhanced web interface"""
        with gr.Blocks(title="Pinnacle AI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 Pinnacle AI - Next Generation")

            with gr.Tabs():
                with gr.Tab("🎯 Smart Task"):
                    self._build_smart_task_tab()

                with gr.Tab("🤖 Agents"):
                    self._build_agents_tab()

                with gr.Tab("📊 Analytics"):
                    self._build_analytics_tab()

                with gr.Tab("🔧 Settings"):
                    self._build_settings_tab()

                with gr.Tab("📚 Templates"):
                    self._build_templates_tab()

        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    def _build_smart_task_tab(self):
        """Build the smart task execution tab"""
        with gr.Row():
            with gr.Column(scale=2):
                self.task_input = gr.Textbox(
                    label="Describe your task",
                    placeholder="What would you like Pinnacle AI to do?",
                    lines=3
                )

                with gr.Accordion("Advanced Options", open=False):
                    self.context_input = gr.Textbox(
                        label="Context (JSON)",
                        placeholder='{"key": "value"}',
                        lines=3
                    )
                    self.task_type = gr.Dropdown(
                        label="Task Type",
                        choices=[
                            "Software Development", "Research", "Creative",
                            "Business", "Scientific", "Personal", "Custom"
                        ],
                        value="Custom"
                    )
                    self.complexity = gr.Slider(
                        label="Complexity Level",
                        minimum=1, maximum=5, value=3, step=1
                    )

                self.execute_btn = gr.Button("Execute Task", variant="primary")
                self.cancel_btn = gr.Button("Cancel", variant="secondary")

            with gr.Column(scale=3):
                self.status_display = gr.Markdown("Ready")
                self.progress = gr.Progress()
                self.result_display = gr.JSON(label="Results")

                if MATPLOTLIB_AVAILABLE:
                    self.visualization = gr.Plot(label="Task Visualization")
                else:
                    self.visualization = gr.Markdown("Visualization requires matplotlib")

                with gr.Row():
                    self.feedback = gr.Radio(
                        label="Feedback",
                        choices=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                        value="⭐⭐⭐"
                    )
                    self.submit_feedback = gr.Button("Submit Feedback")
                    self.save_template = gr.Button("Save as Template")

        # Event handlers
        self.execute_btn.click(
            self._execute_task,
            inputs=[self.task_input, self.context_input, self.task_type, self.complexity],
            outputs=[self.status_display, self.result_display, self.visualization]
        )

        self.submit_feedback.click(
            self._submit_feedback,
            inputs=[self.task_input, self.result_display, self.feedback],
            outputs=self.status_display
        )

        self.save_template.click(
            self._save_template,
            inputs=[self.task_input, self.context_input, self.task_type],
            outputs=self.status_display
        )

    def _execute_task(self, task, context, task_type, complexity):
        """Execute a task with enhanced progress tracking"""
        try:
            # Parse context
            try:
                ctx = json.loads(context) if context else {}
            except json.JSONDecodeError:
                ctx = {}

            # Update status
            status = "Analyzing task..."

            # Execute task
            self.current_task = task
            result = self.pinnacle.execute_task(task, ctx)

            # Generate visualization
            if MATPLOTLIB_AVAILABLE:
                visualization = self._generate_visualization(result)
            else:
                visualization = None

            status = "Task completed!"
            return status, result, visualization

        except Exception as e:
            return f"Error: {str(e)}", None, None

    def _generate_visualization(self, result: Dict):
        """Generate visualization for task results"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8, 4))

            if "execution" in result and "execution" in result["execution"]:
                # Agent contributions visualization
                agents = [step["agent"] for step in result["execution"]["execution"]]
                statuses = [1 if step["status"] == "success" else 0 for step in result["execution"]["execution"]]

                ax.bar(agents, statuses)
                ax.set_title("Agent Execution Status")
                ax.set_ylabel("Success (1) / Failure (0)")
                ax.set_xticklabels(agents, rotation=45, ha='right')

            else:
                # Default visualization
                metrics = {
                    'Quality': result.get("evaluation", {}).get("quality", 0),
                    'Efficiency': result.get("evaluation", {}).get("efficiency", 0),
                    'Success': 1 if result.get("evaluation", {}).get("success", False) else 0
                }

                ax.bar(metrics.keys(), metrics.values())
                ax.set_title("Task Performance Metrics")
                ax.set_ylabel("Score")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return None

    def _submit_feedback(self, task, result, feedback):
        """Submit feedback for a task"""
        try:
            rating = len(feedback)
            self.feedback_data.append({
                "task": task,
                "result": result,
                "rating": rating,
                "timestamp": time.time()
            })
            return f"Feedback submitted: {feedback}"
        except Exception as e:
            return f"Error submitting feedback: {str(e)}"

    def _save_template(self, task, context, task_type):
        """Save task as template"""
        try:
            template = {
                "task": task,
                "context": context,
                "task_type": task_type,
                "timestamp": time.time()
            }
            self.task_history.append(template)
            return f"Template saved: {task[:50]}..."
        except Exception as e:
            return f"Error saving template: {str(e)}"

    def _build_agents_tab(self):
        """Build agents management tab"""
        gr.Markdown("### Agent Status and Management")
        agent_status = gr.Markdown("Agent status will be displayed here")
        refresh_btn = gr.Button("Refresh Status")
        
        refresh_btn.click(
            fn=lambda: "Agent status refreshed",
            outputs=agent_status
        )

    def _build_analytics_tab(self):
        """Build analytics tab"""
        gr.Markdown("### System Analytics")
        analytics_display = gr.Markdown("Analytics will be displayed here")

    def _build_settings_tab(self):
        """Build settings tab"""
        gr.Markdown("### System Settings")
        settings_display = gr.Markdown("Settings will be displayed here")

    def _build_templates_tab(self):
        """Build templates tab"""
        gr.Markdown("### Task Templates")
        templates_display = gr.Markdown("Templates will be displayed here")


if __name__ == "__main__":
    ui = EnhancedWebUI()
    ui.launch()

