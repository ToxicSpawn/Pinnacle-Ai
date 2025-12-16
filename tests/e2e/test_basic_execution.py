"""
End-to-end tests for basic execution
"""

import pytest
from main import PinnacleAI

@pytest.mark.skip(reason="Requires full system setup with API keys")
def test_basic_task_execution():
    """Test basic task execution."""
    pinnacle = PinnacleAI()
    result = pinnacle.execute_task("Test task")
    assert "task" in result
    assert "evaluation" in result

