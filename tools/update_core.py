"""Export an explicitly reviewed model core without copying repository history."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'tools/core_files.json'
STATE = ROOT / 'tools/core_export.json'
FORBIDDEN = (
    re.compile(r'\b[A-Za-z]:[\\/](?:Users|home|Documents)[\\/]', re.I),
    re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|ya29\.[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b'),
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
)
TEST_EXPORTS = {'load_test_data_only', 'load_test_csv', '_extract_close',
                'find_results_path', 'load_results', 'save_results', 'build_rf_test_summary'}


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or '..' in path.parts or str(path) != name
            or any(part.startswith('.') for part in path.parts)):
        raise ValueError(f'Invalid export path: {name!r}')
    if path.suffix != '.py' or any(part in {'tests', 'scripts', '__pycache__'} for part in path.parts):
        raise ValueError(f'Outside reviewed Python core: {name}')
    if path.name.startswith(('test_', 'check_', 'verify_')) or '_test' in path.stem:
        raise ValueError(f'Test or verification file excluded: {name}')
    return name


def transform(name: str, text: str) -> str:
    """Remove test-only re-exports from public training data loaders."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if name in {'rf_model/common/data_utils.py', 'xgb_model/common/data_utils.py'}:
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        for node in reversed(tree.body):
            if isinstance(node, ast.ImportFrom) and node.module == 'test_data_utils':
                del lines[node.lineno - 1:node.end_lineno]
        text = ''.join(lines)
        if name.startswith('rf_model/'):
            for exported in TEST_EXPORTS:
                text = re.sub(r"(?m)^\s*'" + re.escape(exported) + r"',\s*\n", '', text)
            text = text.replace('Re-exports flattening, target scaling, and test data utilities from modular sub-packages.',
                                'Re-exports flattening and target scaling utilities.')
        else:
            text = text.replace('# Test data loading delegated to test_data_utils\n', '')
    # These are descriptive references to a former checkout name, not import names.
    text = text.replace('at claude_xgb root', 'at the repository root')
    text = text.replace('# claude_xgb/', '# repository root')
    text = text.replace('repository root claude_xgb/', 'repository root')
    return text.rstrip() + '\n'


def audit(name: str, text: str) -> None:
    """Reject common private paths and credential formats before any write."""
    if '\x00' in text:
        raise ValueError(f'NUL byte: {name}')
    for number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in FORBIDDEN):
            raise ValueError(f'Privacy review required: {name}:{number}; content was not printed')
    ast.parse(text, filename=name)


def local_dependencies(source: Path, name: str, text: str) -> set[str]:
    """Find repository-local static imports in each training module's search context."""
    importer = source / name
    family = PurePosixPath(name).parts[0]
    found = set()

    def resolve(parts):
        candidate = source.joinpath(*parts)
        for path in (candidate.with_suffix('.py'), candidate / '__init__.py'):
            if path.is_file():
                return path.relative_to(source).as_posix()

    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imports = [(item.name, 0, []) for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [(node.module or '', node.level, [a.name for a in node.names])]
        else:
            continue
        for module, level, aliases in imports:
            if level:
                base = importer.parent
                for _ in range(level - 1):
                    base = base.parent
                bases = [base.relative_to(source).parts]
            else:
                bases = [(family,), ()]
            tail = tuple(module.split('.')) if module else ()
            for base in bases:
                parts = base + tail
                result = resolve(parts)
                if result:
                    found.add(result)
                    for alias in aliases:
                        child = resolve(parts + (alias,))
                        if child:
                            found.add(child)
                    break
    for parent in importer.parents:
        if parent == source:
            break
        init = parent / '__init__.py'
        if init.is_file():
            found.add(init.relative_to(source).as_posix())
    return found


def prepare(source: Path, files: list[str]) -> dict[str, bytes]:
    """Read only allowlisted files and require review for newly introduced dependencies."""
    prepared = {}
    allowed = set(files)
    for name in files:
        safe_name(name)
        path = source / name
        if not path.resolve().is_relative_to(source) or path.is_symlink():
            raise ValueError(f'Source path escapes export root: {name}')
        text = transform(name, path.read_text(encoding='utf-8-sig'))
        audit(name, text)
        missing = local_dependencies(source, name, text) - allowed
        if missing:
            raise ValueError(f'Unreviewed local dependencies in {name}: {sorted(missing)}')
        prepared[name] = text.encode('utf-8')
    return prepared


def git(*args: str) -> str:
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                            text=True, encoding='utf-8', check=True)
    return result.stdout.strip()


def publish(files: list[str], repository: str) -> None:
    """Publish only reviewed updates to this public repository."""
    actual = git('remote', 'get-url', 'origin').removesuffix('.git')
    if actual != repository:
        raise ValueError('origin differs from the configured public repository')
    if git('diff', '--cached', '--name-only'):
        raise ValueError('The staging area is not empty; review it before publishing')
    git('add', '--', *files, 'tools/core_export.json')
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', 'Update reviewed model core')
        git('push', 'origin', 'HEAD')
    else:
        print('No changes to publish.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Read-only research checkout')
    parser.add_argument('--apply', action='store_true', help='Apply reviewed updates locally')
    parser.add_argument('--publish', action='store_true', help='Apply, commit and push to the configured public repository')
    args = parser.parse_args()
    try:
        source = Path(args.source).resolve()
        if not source.is_dir() or source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
            raise ValueError('Source and public checkout must be separate directories')
        policy = json.loads(POLICY.read_text(encoding='utf-8'))
        previous = json.loads(STATE.read_text(encoding='utf-8'))
        files = policy['files']
        if len(files) != len(set(files)) or set(files) != set(previous['files']):
            raise ValueError('Export policy and previous manifest do not agree')
        prepared = prepare(source, files)
        changes = []
        for name, data in prepared.items():
            target = ROOT / name
            if target.is_symlink() or not target.resolve().is_relative_to(ROOT):
                raise ValueError(f'Public destination escapes checkout: {name}')
            current = target.read_bytes() if target.is_file() else None
            old_hash = previous['files'][name]
            if current is not None and checksum(current) not in {old_hash, checksum(data)}:
                raise ValueError(f'Public file has independent edits: {name}; merge explicitly')
            if current != data:
                changes.append(name)
        for name in changes:
            print(name)
        print(f'Reviewed files: {len(files)}; changed: {len(changes)}')
        if not (args.apply or args.publish):
            print('Preview only; use --apply or --publish after reviewing changes.')
            return 0
        for name in changes:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(prepared[name])
        state = {'schema': 1, 'files': {name: checksum(data) for name, data in prepared.items()}}
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if args.publish:
            publish(files, policy['repository'])
        return 0
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f'Export stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
