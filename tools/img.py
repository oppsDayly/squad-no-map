#!/usr/bin/env python3
from pathlib import Path
import re
import sys

BYTES_PER_LINE = 12
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HEADER = SCRIPT_DIR.parent / "src/filters/occluder_builtin.h"


def format_bytes(data: bytes, indent: str) -> str:
    if not data:
        return indent
    lines = []
    for index, byte in enumerate(data):
        if index % BYTES_PER_LINE == 0:
            lines.append([])
        lines[-1].append(f"0x{byte:02x}")
    return "\n".join(indent + ",".join(line) + "," for line in lines)


def replace_array(source: str, var_name: str, data: bytes) -> str:
    pattern = re.compile(
        rf"(static\s+const\s+unsigned\s+char\s+{var_name}\s*\[\]\s*=\s*\{{)(.*?)(\}}\s*;)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        available = sorted(set(re.findall(r'static\s+const\s+unsigned\s+char\s+(\w+)\s*\[\]', source)))
        raise RuntimeError(
            f"Unable to locate array for {var_name}. Available arrays: {available[:5]}..."
        )
    prefix, old_body, suffix = match.groups()
    indent_match = re.search(r"\n(\s*)0x", old_body)
    indent = indent_match.group(1) if indent_match else "    "
    body = format_bytes(data, indent)
    new_body = "\n" + body + "\n"
    return source[: match.start()] + prefix + new_body + suffix + source[match.end():]


def update_filename_strings(source: str, suffixes: list[str]) -> str:
    """Update embedded_occluder_roi{n}.ext filenames to match provided image extensions."""
    out = source
    for idx, ext in enumerate(suffixes):
        pattern = re.compile(rf"(embedded_occluder_roi{idx + 1})\.[A-Za-z0-9]+")
        replacement = rf"\1.{ext}"
        out, n = pattern.subn(replacement, out, count=1)
        if n == 0:
            raise RuntimeError(f"Unable to update filename for ROI{idx + 1}")
    return out


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print("Usage: python tools/img.py <roi1_image> <roi2_image> <roi3_image> [header_path]", file=sys.stderr)
        return 1

    roi_paths = [Path(p).resolve() for p in sys.argv[1:4]]
    header_path = Path(sys.argv[4]).resolve() if len(sys.argv) == 5 else DEFAULT_HEADER

    for path in roi_paths:
        if not path.is_file():
            print(f"Image file not found: {path}", file=sys.stderr)
            return 1
    if not header_path.is_file():
        print(f"Header file not found: {header_path}", file=sys.stderr)
        return 1

    suffixes: list[str] = []
    for p in roi_paths:
        ext = p.suffix.lower().lstrip(".")
        if not ext:
            print(f"Image has no extension: {p}", file=sys.stderr)
            return 1
        suffixes.append(ext)

    roi_data = [p.read_bytes() for p in roi_paths]
    var_names = [
        "g_builtin_occluder_roi1",
        "g_builtin_occluder_roi2",
        "g_builtin_occluder_roi3",
    ]

    header_text = header_path.read_text(encoding="utf-8")
    triplets = list(zip(var_names, roi_paths, roi_data))
    for var_name, _, data in triplets:
        header_text = replace_array(header_text, var_name, data)
    header_text = update_filename_strings(header_text, suffixes)
    header_path.write_text(header_text, encoding="utf-8")

    for var_name, path, data in triplets:
        print(f"{var_name} <- {path.name} ({len(data)} bytes)", file=sys.stderr)
    print(f"Updated {header_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
