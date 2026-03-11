import pytest
from unittest.mock import MagicMock
from agentic_router import AgenticRouter

def test_agentic_router_initialization():
    mock_llm = MagicMock()
    router = AgenticRouter(llm=mock_llm)
    assert router.llm == mock_llm

def test_custom_classification_exact():
    mock_llm = MagicMock()
    # Mock the LLM to return exactly the "Combat" keyword
    mock_llm.complete.return_value = "Combat"
    
    router = AgenticRouter(llm=mock_llm)
    categories = ["Peaceful", "Combat", "Tavern"]
    
    result = router.custom_classification(
        query="The goblins attack!",
        categories=categories,
        prompt_template="{category_list} - {query}",
        default="Unknown"
    )
    
    assert result == "Combat"
    mock_llm.complete.assert_called_once()

def test_custom_classification_fallback():
    mock_llm = MagicMock()
    # Mock the LLM to return garbage
    mock_llm.complete.return_value = "Gobbledygook"
    
    router = AgenticRouter(llm=mock_llm)
    categories = ["Peaceful", "Combat", "Tavern"]
    
    result = router.custom_classification(
        query="The goblins attack!",
        categories=categories,
        prompt_template="{category_list} - {query}",
        default="Unknown"
    )
    
    # Should safely fallback to default since Gobbledygook is not in categories
    assert result == "Unknown"

def test_dynamic_route_query():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Billing"
    
    router = AgenticRouter(llm=mock_llm)
    
    topology = {
        "Tech": "Bug issues.",
        "Billing": "Money issues."
    }
    
    route = router.route_query("Refund me", topology=topology)
    assert route == "Billing"
