"""Create a clean, reproducible code.zip without local state or secrets."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE = ("code", "README.md", "ARCHITECTURE.md", ".design.md", "DECISIONS.md",
           "BACKLOG.md", "SUBMISSION.md", "requirements.txt", ".env.example", "problem_statement.md",
           "scripts", "tests")
EXCLUDE_PARTS = {".venv", ".router-cache", "__pycache__", ".git", "dataset", ".env"}


def should_include(path: Path) -> bool:
    return not any(part in EXCLUDE_PARTS for part in path.parts) and path.suffix != ".pyc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="code.zip")
    args = parser.parse_args()
    root, destination = Path.cwd(), Path(args.destination)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name in INCLUDE:
            item = root / name
            if item.is_file(): archive.write(item, item.relative_to(root))
            elif item.is_dir():
                for path in item.rglob("*"):
                    if path.is_file() and should_include(path.relative_to(root)):
                        archive.write(path, path.relative_to(root))
    print("wrote %s" % destination)


if __name__ == "__main__":
    main()
