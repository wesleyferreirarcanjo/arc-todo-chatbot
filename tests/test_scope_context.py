from app.graph.scope_context import format_scope_context_text


def test_format_scope_context_text_includes_names():
    text = format_scope_context_text(
        {
            "organization_id": "org-1",
            "organization_name": "Acme",
            "project_id": "proj-1",
            "project_name": "Arc Todo",
            "project_description": "Task manager",
        }
    )
    assert 'organization_name="Acme"' in text
    assert 'project_name="Arc Todo"' in text
    assert 'project_description="Task manager"' in text
    assert "organization_id=org-1" in text
