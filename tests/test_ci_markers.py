"""Verify that slow-marked tests exist and the marker is registered."""

import pytest


class TestSlowMarker:
    def test_slow_marker_registered(self, pytestconfig):
        markers = {m.name for m in pytestconfig.getini("markers")
                   if hasattr(m, "name")}
        # markers ini returns strings like "slow: description"
        marker_names = set()
        for m in pytestconfig.getini("markers"):
            name = m.split(":")[0].strip() if isinstance(m, str) else getattr(m, "name", "")
            marker_names.add(name)
        assert "slow" in marker_names

    def test_known_slow_tests_have_marker(self):
        """Confirm @pytest.mark.slow is present on the four flaky tests."""
        import ast
        import pathlib

        slow_targets = {
            "tests/test_stages_47_51.py": "test_permutation_path_no_model_attrs",
            "tests/test_stages_70_74.py": "test_best_agent_returns_agent",
            "tests/test_stages_30_35.py": "test_plan_linear_dependency_chain",
            "tests/test_stages_42_46.py": "test_step_updates_when_enough_data",
        }

        root = pathlib.Path(__file__).parent.parent
        for rel_path, fn_name in slow_targets.items():
            src = (root / rel_path).read_text()
            tree = ast.parse(src)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    for dec in node.decorator_list:
                        dec_str = ast.unparse(dec)
                        if "slow" in dec_str:
                            found = True
            assert found, f"@pytest.mark.slow missing on {fn_name} in {rel_path}"
