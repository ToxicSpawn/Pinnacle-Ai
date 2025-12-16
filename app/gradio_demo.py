#!/usr/bin/env python3
"""
Gradio Demo - Pinnacle AI

Web-based interface for Pinnacle AI using Gradio.
"""

import sys
from pathlib import Path
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PinnacleAI

# Initialize Pinnacle AI
try:
    pinnacle = PinnacleAI()
    print("✓ Pinnacle AI initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    pinnacle = None

def execute_task(task: str, context: str = "") -> str:
    """Execute a task and return formatted result."""
    if not pinnacle:
        return "Error: Pinnacle AI not initialized. Check configuration."
    
    if not task.strip():
        return "Please enter a task."
    
    try:
        # Parse context if provided
        task_context = {}
        if context.strip():
            # Simple context parsing (can be enhanced)
            task_context = {"context": context}
        
        # Execute task
        result = pinnacle.execute_task(task, task_context)
        
        # Format result
        output = []
        output.append(f"**Task:** {result.get('task', task)}")
        output.append(f"**Success:** {'✓' if result.get('evaluation', {}).get('success') else '✗'}")
        
        if result.get('evaluation'):
            eval_data = result['evaluation']
            output.append(f"**Quality:** {eval_data.get('quality', 0):.1%}")
            output.append(f"**Efficiency:** {eval_data.get('efficiency', 0):.1%}")
        
        if result.get('execution', {}).get('execution'):
            output.append("\n**Agents Used:**")
            for exec_item in result['execution']['execution']:
                agent = exec_item.get('agent', 'unknown')
                status = "✓" if "error" not in exec_item.get('result', {}) else "✗"
                output.append(f"- {agent}: {status}")
        
        if result.get('result'):
            output.append("\n**Result:**")
            result_data = result['result']
            if isinstance(result_data, dict):
                for key, value in result_data.items():
                    if isinstance(value, (str, int, float)):
                        output.append(f"- {key}: {value}")
            else:
                output.append(str(result_data))
        
        if "error" in result:
            output.append(f"\n**Error:** {result['error']}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error executing task: {str(e)}"

def trigger_improvement() -> str:
    """Trigger self-improvement cycle."""
    if not pinnacle:
        return "Error: Pinnacle AI not initialized."
    
    try:
        improvements = pinnacle.orchestrator.improve_system()
        output = ["**Self-Improvement Results:**\n"]
        for component, result in improvements.items():
            status = result.get("status", "unknown")
            output.append(f"- {component}: {status}")
        return "\n".join(output)
    except Exception as e:
        return f"Error during improvement: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Pinnacle AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Pinnacle AI - Interactive Interface
    
    The Absolute Pinnacle of Artificial Intelligence
    
    Enter a task below and let Pinnacle AI handle it using its specialized agents.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            task_input = gr.Textbox(
                label="Task",
                placeholder="Enter your task here... (e.g., 'Write a Python function to sort a list')",
                lines=3
            )
            context_input = gr.Textbox(
                label="Context (Optional)",
                placeholder="Additional context for the task...",
                lines=2
            )
            execute_btn = gr.Button("Execute Task", variant="primary")
        
        with gr.Column(scale=1):
            improve_btn = gr.Button("Trigger Self-Improvement", variant="secondary")
    
    output = gr.Markdown(label="Result")
    
    # Event handlers
    execute_btn.click(
        fn=execute_task,
        inputs=[task_input, context_input],
        outputs=output
    )
    
    improve_btn.click(
        fn=trigger_improvement,
        outputs=output
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["Write a Python function to calculate fibonacci numbers"],
            ["Research the latest developments in quantum computing"],
            ["Create a short story about space exploration"],
            ["Plan a software development project"],
            ["Analyze the philosophical implications of AI"],
        ],
        inputs=task_input
    )
    
    gr.Markdown("""
    ## Tips
    
    - Be specific in your task descriptions
    - Use the context field for additional information
    - Try different types of tasks (coding, research, creative, etc.)
    - Use "Trigger Self-Improvement" to enhance the system
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

