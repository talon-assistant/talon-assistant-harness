import json

from core.resume_builder import ResumeLibrary, get_role_markers, get_resume_template_path
from scripts.check_depersonalized import scan_text


def _settings(tmp_path, payload):
    (tmp_path / "settings.example.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_resume_parser_uses_configured_sections_not_embedded_employers(
    tmp_path, monkeypatch,
):
    _settings(tmp_path, {
        "resume": {
            "sections": [
                {"header": "Example Industries", "slug": "recent_role", "cap": 5},
                {"header": "Selected Projects", "slug": "projects", "cap": 2},
            ]
        }
    })
    monkeypatch.setenv("TALON_CONFIG_DIR", str(tmp_path))
    library_path = tmp_path / "library.md"
    library_path.write_text(
        "## Example Industries\n\n- Improved service reliability.\n\n"
        "## Selected Projects\n\n- Built an automation tool.\n",
        encoding="utf-8",
    )

    library = ResumeLibrary(library_path).parse()

    assert list(library.sections) == ["recent_role", "projects"]
    assert library.caps == {"recent_role": 5, "projects": 2}
    assert library.get("recent_role").bullets == ["Improved service reliability."]


def test_resume_template_and_role_markers_come_from_settings(tmp_path, monkeypatch):
    _settings(tmp_path, {
        "resume": {
            "template_path": str(tmp_path / "local-template.docx"),
            "role_markers": [
                {"slug": "recent_role", "marker": "Example Industries"},
            ],
        }
    })
    monkeypatch.setenv("TALON_CONFIG_DIR", str(tmp_path))

    assert get_resume_template_path() == tmp_path / "local-template.docx"
    assert get_role_markers() == [("recent_role", "Example Industries")]


def test_depersonalization_scanner_flags_local_identity_artifacts():
    windows_home = chr(92).join(("C:", "Users", "localperson", "project"))
    assert "absolute user-home path" in scan_text(
        "docs/handoff.md", f"Runtime: {windows_home}"
    )
    assert "non-example email address" in scan_text(
        "docs/handoff.md", "Contact: person@private-domain.test"
    )
    assert "phone-number pattern" in scan_text(
        "docs/handoff.md", "Call (212) 555-0199"
    )


def test_public_settings_profile_is_blank_and_portable():
    settings = json.loads(
        open("config/settings.example.json", encoding="utf-8").read()
    )
    assert all(not value for key, value in settings["user_profile"].items()
               if not key.startswith("_"))
    serialized = json.dumps(settings)
    assert "OneDrive" not in serialized
    user_home_prefix = chr(92).join(("C:", "Users", ""))
    assert user_home_prefix not in serialized
