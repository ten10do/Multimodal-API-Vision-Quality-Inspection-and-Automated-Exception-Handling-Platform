"""Validate tracked Markdown links, structure, and Mermaid blocks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>\n]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
REFERENCE_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(?P<target><[^>]+>|\S+)", re.MULTILINE)
HTML_LINK_RE = re.compile(r"(?:href|src)=[\"'](?P<target>[^\"']+)[\"']", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:")


def markdown_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(ROOT / Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item)


def exact_case_exists(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return False

    cursor = ROOT
    for part in relative.parts:
        if not cursor.is_dir() or part not in {entry.name for entry in cursor.iterdir()}:
            return False
        cursor /= part
    return cursor.exists()


def resolve_local_link(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target or target.startswith("#") or target.lower().startswith(EXTERNAL_PREFIXES):
        return None

    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    if path_text.startswith("/"):
        return (ROOT / path_text.lstrip("/")).resolve()
    return (source.parent / path_text).resolve()


def inspect_markdown(path: Path) -> tuple[str, list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    prose: list[str] = []
    mermaid_blocks: list[str] = []
    fence_char = ""
    fence_length = 0
    fence_line = 0
    language = ""
    block: list[str] = []

    if "\ufffd" in text:
        errors.append(f"{path.relative_to(ROOT)}: contains Unicode replacement characters")

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.endswith((" ", "\t")):
            errors.append(f"{path.relative_to(ROOT)}:{line_number}: trailing whitespace")

        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char = marker[0]
                fence_length = len(marker)
                fence_line = line_number
                language = fence.group(2).strip().lower()
                block = []
                continue
            if marker[0] == fence_char and len(marker) >= fence_length:
                if language == "mermaid":
                    mermaid_blocks.append("\n".join(block).strip() + "\n")
                fence_char = ""
                fence_length = 0
                language = ""
                block = []
                continue

        if fence_char:
            block.append(line)
        else:
            prose.append(INLINE_CODE_RE.sub("", line))

    if fence_char:
        errors.append(f"{path.relative_to(ROOT)}:{fence_line}: unclosed code fence")

    return "\n".join(prose), mermaid_blocks, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-mermaid", type=Path)
    args = parser.parse_args()

    files = markdown_files()
    errors: list[str] = []
    diagrams: list[tuple[Path, int, str]] = []
    local_links = 0

    for path in files:
        try:
            prose, blocks, file_errors = inspect_markdown(path)
        except UnicodeDecodeError as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid UTF-8: {exc}")
            continue

        errors.extend(file_errors)
        diagrams.extend((path, index, block) for index, block in enumerate(blocks, start=1))
        matches = list(INLINE_LINK_RE.finditer(prose))
        matches += list(REFERENCE_RE.finditer(prose))
        matches += list(HTML_LINK_RE.finditer(prose))
        for match in matches:
            target = resolve_local_link(path, match.group("target"))
            if target is None:
                continue
            local_links += 1
            if not exact_case_exists(target):
                errors.append(
                    f"{path.relative_to(ROOT)}: broken or case-mismatched link: "
                    f"{match.group('target')}"
                )

    if args.extract_mermaid:
        output = args.extract_mermaid
        if not output.is_absolute():
            output = ROOT / output
        output.mkdir(parents=True, exist_ok=True)
        for old_file in output.glob("*.mmd"):
            old_file.unlink()
        for path, index, content in diagrams:
            relative = path.relative_to(ROOT).as_posix().replace("/", "__")
            (output / f"{relative}__{index}.mmd").write_text(content, encoding="utf-8")

    print(f"markdown_files={len(files)}")
    print(f"local_links={local_links}")
    print(f"mermaid_blocks={len(diagrams)}")
    print(f"errors={len(errors)}")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
