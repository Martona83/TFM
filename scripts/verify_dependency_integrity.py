from pathlib import Path
import hashlib

LOCKFILE_POLICY = {
    "strategy": "uv",
    "required_lockfiles": ["requirements.txt", "src/requirements.txt", "src/requirements_core.txt", "src/requirements_fairness.txt"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    missing = []
    for p in LOCKFILE_POLICY['required_lockfiles']:
        if not Path(p).exists():
            missing.append(p)
    if missing:
        raise SystemExit(f"Missing dependency lock/snapshot files: {missing}")
    for p in LOCKFILE_POLICY['required_lockfiles']:
        print(f"{p}: {sha256(Path(p))}")
    print(f"lockfile_strategy={LOCKFILE_POLICY['strategy']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
