"""Planner tests for the retrieval routing heuristics.

These checks make sure the keyword-based planner still routes obvious
authentication questions to the expected tools.
"""

from agents.planner import PlannerAgent


def test_planner_auth():
    plan = PlannerAgent().plan("Explain authentication and JWT")
    assert plan.steps
    assert plan.steps[0].tool_name == "search_readme"
    assert plan.steps[0].query.startswith("authentication login jwt token session refresh")
    assert any(step.tool_name == "search_routes" and "middleware" in step.query for step in plan.steps)
