from pathlib import Path
from subprocess import run


def test_repository_contains_no_competition_payload() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = run(
        ["git", "ls-files", "--cached", "--", "case-set.json", "inputs", "outputs"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert not any(path == "case-set.json" or path.endswith(".json") for path in tracked)
    forbidden = {"oracles", "reference-outputs", "private-partitions.json", "mcp-access.json"}
    assert not any(path.name in forbidden for path in root.rglob("*"))


def test_example_environment_has_no_real_key() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / ".env.example").read_text(encoding="utf-8")
    assert "sk-team-replace_me" in content
    assert content.count("sk-team-") == 1
