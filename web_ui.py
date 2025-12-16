"""
Gradio Web Interface for Pinnacle AI
"""

import gradio as gr
from src_main import PinnacleAI

# Initialize Pinnacle AI
pinnacle = PinnacleAI()

def process_task(task, history):
    """Process a task and return the result."""
    if not task or not task.strip():
        return "Please enter a task to execute."
    
    result = pinnacle.execute_task(task)
    return format_result(result)

def format_result(result):
    """Format the result for display."""
    output = f"### Task: {result['task']}\n\n"
    output += f"**Success:** {'✅' if result['evaluation']['success'] else '❌'}\n\n"
    
    if "error" in result:
        output += f"**Error:** {result['error']}\n"
        return output
    
    output += f"**Quality:** {result['evaluation']['quality']:.1%}\n"
    output += f"**Efficiency:** {result['evaluation']['efficiency']:.1%}\n\n"
    
    if result["execution"]["execution"]:
        output += "### Agent Contributions:\n"
        for execution in result["execution"]["execution"]:
            status = "✅" if execution["status"] == "success" else "❌"
            output += f"- **{execution['agent']}**: {status}\n"
            if "error" in execution.get("result", {}):
                output += f"  Error: {execution['result']['error']}\n"
    
    if result.get("learning", {}).get("learning_outcomes"):
        output += "\n### Learning Outcomes:\n"
        for outcome_type, outcome in result["learning"]["learning_outcomes"].items():
            if isinstance(outcome, (list, dict)):
                output += f"- {outcome_type}: {len(outcome)} items\n"
            else:
                output += f"- {outcome_type}: {outcome}\n"
    
    return output

def respond(message, chat_history):
    """Respond to a message in chat mode."""
    if not message or not message.strip():
        return "", chat_history
    
    result = pinnacle.execute_task(message)
    formatted = format_result(result)
    chat_history.append((message, formatted))
    return "", chat_history

def launch_web_ui():
    """Launch the Gradio web interface."""
    with gr.Blocks(title="Pinnacle AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Pinnacle AI
        ## The Absolute Pinnacle of Artificial Intelligence
        
        Pinnacle AI is an advanced AGI system with:
        - **Multi-Agent Coordination**: Specialized agents working together
        - **Neurosymbolic Reasoning**: Combining neural and symbolic AI
        - **Self-Improvement**: Continuous learning and optimization
        - **Hyper-Modal Processing**: Text, image, audio, and more
        """)
        
        with gr.Tab("Task Execution"):
            gr.Markdown("### Execute a single task")
            task_input = gr.Textbox(
                label="Enter your task",
                placeholder="Write a Python script to analyze data...",
                lines=3
            )
            submit_btn = gr.Button("Execute Task", variant="primary")
            output = gr.Markdown(label="Results")
            
            submit_btn.click(
                fn=process_task,
                inputs=[task_input],
                outputs=output
            )
            
            task_input.submit(
                fn=process_task,
                inputs=[task_input],
                outputs=output
            )
        
        with gr.Tab("Examples"):
            gr.Markdown("### Try these example tasks")
            gr.Examples(
                examples=[
                    ["Write a Python web application using FastAPI"],
                    ["Research the latest advancements in quantum computing"],
                    ["Create a business plan for an AI startup"],
                    ["Design a scientific experiment to test a new hypothesis"],
                    ["Compose a short story about artificial consciousness"],
                    ["Plan a complex software development project"],
                    ["Analyze the philosophical implications of AGI"],
                    ["Write a function to calculate Fibonacci numbers with memoization"]
                ],
                inputs=task_input,
                label="Example Tasks"
            )
        
        with gr.Tab("Interactive Chat"):
            gr.Markdown("### Chat with Pinnacle AI")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True
            )
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                show_label=False
            )
            clear = gr.Button("Clear Chat", variant="secondary")
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("System Status"):
            gr.Markdown("### System Information")
            status_info = gr.Markdown("""
            **Available Agents:**
            - Planner: Strategic planning and task decomposition
            - Researcher: Information gathering and analysis
            - Coder: Programming and code generation
            - Creative: Content creation and artistic tasks
            - Robotic: Physical world interaction
            - Scientist: Scientific research and experimentation
            - Philosopher: Deep reasoning and ethical analysis
            - Meta Agent: Coordination and orchestration
            
            **Capabilities:**
            - Multi-agent coordination
            - Neurosymbolic reasoning
            - Self-improvement
            - Hyper-modal processing
            - Continuous learning
            """)
            
            refresh_btn = gr.Button("Refresh Status")
            refresh_btn.click(
                fn=lambda: "Status refreshed at " + str(__import__("datetime").datetime.now()),
                outputs=status_info
            )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    launch_web_ui()

