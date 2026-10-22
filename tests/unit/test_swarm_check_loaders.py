"""Policy: every dynamic-import call site in vibe_quant/ is modelled by the gate.

scripts/agents/swarm_check.py selects tests through the import graph. A module loaded by
directory scan or a computed name (pkgutil.iter_modules, importlib.import_module, __import__,
spec_from_file_location, spec_from_loader, importlib.reload) has no import edge, so each such
file must be a DIRECTORY_LOADERS loader or a LOADER_EXEMPTIONS entry with a reason. A new
loader otherwise reopens the gap the round-1/2 reviews found (rams.py, research sources);
the gate itself also runs the full suite while one is unmodelled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "swarm_check.py"
if "swarm_check" in sys.modules:
    swarm_check = sys.modules["swarm_check"]
else:
    _spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    swarm_check = importlib.util.module_from_spec(_spec)
    sys.modules["swarm_check"] = swarm_check
    _spec.loader.exec_module(swarm_check)


def test_every_dynamic_import_site_is_modelled() -> None:
    sites = swarm_check.loader_call_sites(_ROOT)
    assert sites, "the scan found no call site at all: the regex is broken"
    assert swarm_check.unmodelled_loaders(_ROOT) == []


def test_exemptions_have_reasons_and_still_exist() -> None:
    sites = swarm_check.loader_call_sites(_ROOT)
    for path, reason in swarm_check.LOADER_EXEMPTIONS.items():
        assert len(reason.split()) >= 4, path
        assert path in sites, f"stale exemption {path}: no dynamic-import call site left"


def test_the_scan_sees_each_call_form(tmp_path: Path) -> None:
    pkg = tmp_path / "vibe_quant"
    pkg.mkdir()
    forms = {
        "a.py": "pkgutil.iter_modules(p)\n",
        "b.py": "importlib.import_module(name)\n",
        "c.py": "x = __import__('os')\n",
        "d.py": "importlib.util.spec_from_file_location(n, p)\n",
        "e.py": "importlib.util.spec_from_loader(n, loader=None)\n",
        "f.py": "importlib.reload(m)\n",
        "g.py": 'BANNED = {"__import__", "import_module"}\n',  # a string, no call
    }
    for name, text in forms.items():
        (pkg / name).write_text(text)
    assert sorted(swarm_check.loader_call_sites(tmp_path)) == [
        f"vibe_quant/{n}.py" for n in "abcdef"
    ]
