#!/usr/bin/env python3
"""coupling-guard.py — is the serving profile still coupled to its pinned upstreams?

This recipe's serving profile (`config/cluster.env`) is written against two
pinned public upstream checkouts: the compose/launcher line that supplies the
env shape and the boot machinery, and the TP=3 attention-group pad kit. Every
variable the profile sets is either carried by one of those upstreams, owned
by this recipe's own appliers, or carried by nobody (a reader/library-level
knob). `config/coupling-manifest.json` records which is which.

The guard is the preflight that proves the contract still holds. It checks
that the pinned revisions still exist, that the pinned upstream sources still
carry the variable names this profile depends on, and that this recipe's own
appliers still read the variables they own. A renamed or removed upstream
variable is a **FAIL**, loudly: the profile would boot with a knob that
silently does nothing, which is the exact failure the fixlog records.

"Carried" means carried by something that CONSUMES the variable, not by
something that describes it. A variable that survives only in a
documentation file is a FAIL as well: the upstream renamed the knob and
its docs have not caught up, which is the realistic shape of the drift —
docs are the last thing a project updates.

Three ways to run it:

  * **offline, against a pinned checkout** — no network at all:

        python3 tools/coupling-guard.py \\
            --compose-dir /path/to/pinned/compose-checkout \\
            --padkit-dir /path/to/pinned/padkit-checkout

  * **offline, against a pinned archive** (a tarball of the same rev):

        python3 tools/coupling-guard.py \\
            --compose-archive /path/to/compose-<rev>.tar.gz \\
            --padkit-archive /path/to/padkit-<rev>.tar.gz

  * **online, against the live upstream** — proves the pinned revisions still
    exist and still carry the names:

        python3 tools/coupling-guard.py --remote

The online mode NEVER reports a pass it did not earn. If the network is
unavailable it prints a warning and exits **INCONCLUSIVE** (2), which is
distinct from PASS (0) and from FAIL (1).

Exit codes:
    0  PASS         the profile and the pinned upstreams still agree
    1  FAIL         a variable was renamed or removed, or a pinned source is gone
    2  INCONCLUSIVE the check could not complete (network down, source unreadable)
    3  USAGE        the tool was called with the wrong arguments

`--selftest` proves the FAIL path works: it builds a synthetic upstream tree
in a temporary directory, renames a variable in it, and shows the guard
rejecting it. A guard that cannot fail is not a guard.
"""

# The `apply_synthetic.py` fixture below is created at runtime in a temporary
# directory by --selftest; it is not a shipped file.
# selftest-ignore-ref: apply_synthetic.py

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INCONCLUSIVE = 2
EXIT_USAGE = 3

RAW_BASE = "https://raw.githubusercontent.com"
TARBALL_BASE = "https://codeload.github.com"

RECIPE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = RECIPE_ROOT / "config" / "coupling-manifest.json"
DEFAULT_PROFILE = RECIPE_ROOT / "config" / "cluster.env"

TEXT_SUFFIXES = {".py", ".sh", ".md", ".env", ".yaml", ".yml", ".toml", ".cfg", ".json"}


# --------------------------------------------------------------------------
# report


class Report:
    """Collects failures, warnings and inconclusive notes for one run."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.inconclusive: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def fail_inconclusive(self, msg: str) -> None:
        self.inconclusive.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "Report") -> None:
        self.failures.extend(other.failures)
        self.warnings.extend(other.warnings)
        self.inconclusive.extend(other.inconclusive)


# --------------------------------------------------------------------------
# inputs


def parse_profile(path: Path) -> list[str]:
    """The variable names the profile sets, in file order."""
    names: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name not in names:
            names.append(name)
    return names


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(errors="replace"))
    # The manifest writes its upstream paths as "./<path>", meaning
    # "relative to the upstream checkout root". Strip the marker here so the
    # rest of the tool works with plain relative paths.
    for upstream in manifest.get("upstreams", []):
        files = upstream.get("search_files")
        if isinstance(files, list):
            upstream["search_files"] = [
                f[2:] if f.startswith("./") else f for f in files
            ]
    return manifest


def emit(quiet: bool, *parts: object) -> None:
    """Print unless the caller asked for quiet output."""
    if not quiet:
        print(*parts)


def var_pattern(name: str) -> re.Pattern:
    """A whole-token match for a variable name."""
    return re.compile(r"\b" + re.escape(name) + r"\b")


def upstream_keys(manifest: dict) -> set[str]:
    return {u["key"] for u in manifest["upstreams"]}


def ownership_classes(manifest: dict) -> list[tuple[str, dict]]:
    """The classes that say WHO owns a variable (exactly one per variable)."""
    keys = upstream_keys(manifest)
    return [(n, c) for n, c in manifest["classes"].items() if c.get("carrier") not in keys]


# --------------------------------------------------------------------------
# sources: a local checkout, a local archive, or the live upstream


class Source:
    """A pinned upstream source. Subclasses know how to fetch one file."""

    kind = "source"

    def __init__(self, label: str) -> None:
        self.label = label

    def read(self, rel: str) -> tuple[str | None, str]:
        """(content, error). A missing file is an error, not a silent empty."""
        raise NotImplementedError

    def revision(self) -> tuple[str, str | None, str]:
        """(status, rev, note). status: ok | missing | unreachable | unknown."""
        return "unknown", None, "the rev could not be read from this source"


class LocalCheckout(Source):
    """A checkout directory of the pinned upstream."""

    kind = "checkout"

    def __init__(self, label: str, path: Path, search_files: list[str]) -> None:
        super().__init__(label)
        self.path = path
        self.search_files = search_files
        self.root = self._resolve_root()

    def _resolve_root(self) -> Path:
        if not self.path.is_dir():
            return self.path
        for rel in self.search_files:
            if (self.path / rel).is_file():
                return self.path
        for child in sorted(self.path.iterdir()):
            if not child.is_dir():
                continue
            for rel in self.search_files:
                if (child / rel).is_file():
                    return child
        return self.path

    def read(self, rel: str) -> tuple[str | None, str]:
        for base in (self.root, self.path):
            candidate = base / rel
            if candidate.is_file():
                try:
                    return candidate.read_text(errors="replace"), ""
                except OSError as exc:
                    return None, f"unreadable ({exc.__class__.__name__})"
        return None, "not present in this checkout"

    def revision(self) -> tuple[str, str | None, str]:
        if not (self.root / ".git").exists():
            return "unknown", None, "not a git checkout — the rev could not be read"
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return "unknown", None, f"git unavailable ({exc.__class__.__name__})"
        if proc.returncode != 0:
            return "unknown", None, "git could not resolve HEAD"
        return "ok", proc.stdout.strip(), ""


class LocalArchive(Source):
    """A tarball of the pinned upstream. Read in memory — never extracted."""

    kind = "archive"

    def __init__(self, label: str, path: Path) -> None:
        super().__init__(label)
        self.path = path
        self.contents: dict[str, str] = {}
        self.error = ""
        try:
            with tarfile.open(path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    handle = tf.extractfile(member)
                    if handle is None:
                        continue
                    try:
                        text = handle.read().decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    self.contents[member.name] = text
                    parts = member.name.split("/", 1)
                    if len(parts) == 2:
                        self.contents.setdefault(parts[1], text)
        except (OSError, tarfile.TarError) as exc:
            self.error = f"unreadable archive ({exc.__class__.__name__}: {exc})"

    def read(self, rel: str) -> tuple[str | None, str]:
        if self.error:
            return None, self.error
        if rel in self.contents:
            return self.contents[rel], ""
        suffix = "/" + rel
        for name, text in self.contents.items():
            if name.endswith(suffix):
                return text, ""
        return None, "not present in this archive"

    def revision(self) -> tuple[str, str | None, str]:
        return "unknown", None, "an archive carries no git metadata"


class RemoteSource(Source):
    """The live upstream, at the pinned revision."""

    kind = "remote"

    def __init__(self, label: str, repo: str, rev: str,
                 raw_base: str = RAW_BASE, tar_base: str = TARBALL_BASE,
                 timeout: int = 30) -> None:
        super().__init__(label)
        self.repo = repo
        self.rev = rev
        self.raw_base = raw_base
        self.tar_base = tar_base
        self.timeout = timeout

    def tarball_url(self) -> str:
        return f"{self.tar_base}/{self.repo}/tar.gz/{self.rev}"

    def raw_url(self, rel: str) -> str:
        return f"{self.raw_base}/{self.repo}/{self.rev}/{rel}"

    def read(self, rel: str) -> tuple[str | None, str]:
        status, body, error = http_fetch(self.raw_url(rel), "GET", self.timeout)
        if status == 200 and body is not None:
            return body, ""
        if status == 404:
            return None, "not present at this revision"
        if status is None:
            return None, f"unreachable: the upstream could not be reached ({error})"
        return None, f"unexpected HTTP {status}"

    def revision(self) -> tuple[str, str | None, str]:
        status, _body, error = http_fetch(self.tarball_url(), "HEAD", self.timeout)
        if status == 200:
            return "ok", self.rev, ""
        if status == 404:
            return "missing", None, "the pinned revision no longer resolves (HTTP 404)"
        return "unreachable", None, (
            f"the upstream could not be reached ({error or 'HTTP ' + str(status)})"
        )


def http_fetch(url: str, method: str = "GET", timeout: int = 30) -> tuple[int | None, str | None, str]:
    """(status, body, error). status is None when the request could not complete."""
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace") if method == "GET" else ""
            return resp.status, body, ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, ""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, None, f"{exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------
# checks


def collect_text(source: Source, files: list[str]) -> tuple[list[tuple[str, str]], list[str], bool, set[str]]:
    """(the text of each file that was present, the files that were present,
    whether the source was unreachable, the files that could not be fetched).

    The text is kept PER FILE, not concatenated. A concatenation makes a
    variable count as "carried" if its name appears in ANY ONE file --
    including a documentation file, which is the LAST thing an upstream
    updates when it renames a knob. That is the realistic drift shape:
    the env template, the compose file and the launcher are renamed, the
    docs still say the old name, and a union search reports the variable
    as present. The guard would then hand a reader a false safety
    attestation on exactly the failure it exists to catch.

    A file that could not be FETCHED is kept apart from a file that is
    ABSENT. A 404 means the upstream moved or removed it; a network
    error means we never asked. Only the first is evidence, so only the
    first is a failure.
    """
    chunks: list[tuple[str, str]] = []
    present: list[str] = []
    unfetchable: set[str] = set()
    unreachable = False
    for rel in files:
        content, error = source.read(rel)
        if content is None:
            if error.startswith("unreachable:"):
                unreachable = True
                unfetchable.add(rel)
            continue
        chunks.append((rel, content))
        present.append(rel)
    return chunks, present, unreachable, unfetchable


def is_documentation(rel: str) -> bool:
    """Is this file documentation rather than something that CONSUMES a
    variable?

    A file under a `docs/` directory, or a README, describes the variable;
    it does not read it. Only a non-documentation file can carry a
    variable for the purposes of this check.

    AN ENV TEMPLATE IS NOT DOCUMENTATION, and that is deliberate. An env
    file is consumed by the engine that loads it: the pinned padkit's
    three-node bootstrap hands its assembled env file to `docker compose
    --env-file`, and the template that env file is assembled from is
    what carries the values. A variable named only in that template
    IS carried by the upstream. Counting env templates as documentation
    was considered and rejected: it turns two carried variables
    (`MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`) into false failures, and
    a guard that cries wolf on a true positive gets ignored.
    """
    parts = rel.replace("\\", "/").split("/")
    if any(p == "docs" for p in parts[:-1]):
        return True
    name = parts[-1].lower()
    return name == "readme.md" or name.endswith(".md")


def check_upstream(source: Source, upstream: dict, classes: list[tuple[str, dict]],
                  quiet: bool = False) -> Report:
    """Check one upstream: its revision, its pinned files, its variables."""
    rep = Report()
    key = upstream["key"]
    rev = upstream["rev"]
    files = upstream.get("search_files", [])

    emit(quiet, f"[{key}] {upstream['repo']} @ {rev[:12]}")
    emit(quiet, f"  source: {source.label} ({source.kind})")

    status, got, note = source.revision()
    if status == "ok":
        if source.kind == "remote":
            emit(quiet, "  revision: the pinned revision still resolves")
        elif got and got.lower().startswith(rev.lower()[:12]):
            emit(quiet, f"  revision: checkout HEAD {got[:12]} matches the pinned revision")
        else:
            rep.fail(
                f"[{key}] the checkout is at {(got or '?')[:12]}, not the pinned revision "
                f"{rev[:12]} — validate against the pinned revision, or update the manifest"
            )
            emit(quiet, f"  revision: MISMATCH — checkout HEAD {(got or '?')[:12]} != pinned {rev[:12]}")
    elif status == "missing":
        rep.fail(f"[{key}] {note} — the pinned revision is gone")
        emit(quiet, f"  revision: MISSING — {note}")
    elif status == "unreachable":
        rep.fail_inconclusive(f"[{key}] {note}")
        emit(quiet, f"  revision: UNREACHABLE — {note}")
    else:
        emit(quiet, f"  revision: unverified — {note}")

    chunks, present, unreachable, unfetchable = collect_text(source, files)
    missing_files = [f for f in files if f not in present and f not in unfetchable]
    emit(quiet, f"  pinned source files: {len(present)}/{len(files)} present")
    for rel in sorted(unfetchable):
        # NOT a missing file: we never got to ask. Saying "upstream removed
        # it" here would assert a removal that was never established.
        emit(quiet, f"    NOT FETCHED {rel} (the upstream could not be reached — "
                    "this file was not checked)")
    for rel in missing_files:
        rep.fail(
            f"[{key}] the pinned source file {rel} is missing — upstream moved or removed it "
            "at the pinned revision"
        )
        emit(quiet, f"    MISSING {rel}")

    carried = [(cname, cdef) for cname, cdef in classes if cdef.get("carrier") == key]
    checked = 0
    missing_vars: list[str] = []
    doc_only_vars: list[str] = []

    if unreachable and not present:
        # NOTHING was read. Reporting every variable as MISSING would be a
        # false statement about the upstream -- it would say "upstream
        # renamed these" when the truth is "we could not reach upstream".
        # The verdict is already INCONCLUSIVE from the revision check.
        emit(quiet, "  upstream-carried variables: NOT CHECKED (source unreachable)")
        return rep

    code_chunks = [(rel, text) for rel, text in chunks if not is_documentation(rel)]
    doc_chunks = [(rel, text) for rel, text in chunks if is_documentation(rel)]
    for _cname, cdef in carried:
        for var in cdef["vars"]:
            checked += 1
            pattern = var_pattern(var)
            if any(pattern.search(text) for _rel, text in code_chunks):
                continue
            if any(pattern.search(text) for _rel, text in doc_chunks):
                # The name survives only in documentation. That is not the
                # upstream carrying the variable; it is the docs lagging
                # behind a rename.
                doc_only_vars.append(var)
                rep.fail(
                    f"[{key}] {var} appears only in documentation in the pinned upstream "
                    "sources — the upstream describes it but no longer reads it "
                    "(renamed upstream, docs not yet updated)"
                )
                continue
            missing_vars.append(var)
            rep.fail(
                f"[{key}] {var} no longer appears in the pinned upstream sources "
                "(renamed or removed upstream) — the profile would set a knob that "
                "does nothing"
            )
    for var in doc_only_vars:
        emit(quiet, f"    DOC-ONLY VAR {var}")
    for var in missing_vars:
        emit(quiet, f"    MISSING VAR {var}")
    if checked:
        ok = checked - len(missing_vars) - len(doc_only_vars)
        emit(quiet, f"  upstream-carried variables: {ok}/{checked} still present"
                    + (f" ({len(doc_only_vars)} documentation-only, "
                       f"{len(missing_vars)} gone)" if doc_only_vars or missing_vars else ""))
    return rep


def recipe_source_chunks(recipe_root: Path, dirs: list[str]) -> list[tuple[str, str]]:
    """The recipe's own sources, as (path, text) pairs -- NOT one blob.

    The path is kept because "is this variable still read?" depends on
    WHICH file reads it. A union of every file's text answers the wrong
    question: `patches/README.md` names every recipe-owned variable, so
    a union reports all of them as present even after every applier that
    owned them has been deleted. The check would pass while testing
    nothing, and a check that cannot fail is worse than no check.
    """
    chunks: list[tuple[str, str]] = []
    for sub in dirs:
        base = recipe_root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                try:
                    rel = str(path.relative_to(recipe_root))
                except ValueError:
                    rel = str(path)
                try:
                    chunks.append((rel, path.read_text(errors="replace")))
                except OSError:
                    continue
    return chunks


def check_recipe(recipe_root: Path, classes: list[tuple[str, dict]], dirs: list[str],
                 quiet: bool = False) -> Report:
    """Check the variables this recipe's own appliers read.

    A recipe-owned variable is read by ONE declared owner file (the manifest
    names it). The check therefore asks "does THAT file still read it?", not
    "does the name appear anywhere in the tree?": a sibling applier, a probe's
    docstring or a comment that happens to mention the knob is not a reader.
    A union over every file would keep reporting the variable as present
    after its applier was deleted, and a check that cannot fail is worse
    than no check. When the manifest declares no owner for a variable, the
    union is still used as a fallback.
    """
    rep = Report()
    emit(quiet, "[recipe] " + ", ".join(d + "/" for d in dirs))
    chunks = recipe_source_chunks(recipe_root, dirs)
    code_chunks = [(rel, text) for rel, text in chunks if not is_documentation(rel)]
    doc_chunks = [(rel, text) for rel, text in chunks if is_documentation(rel)]
    code_by_rel = {rel: text for rel, text in code_chunks}
    checked = 0
    missing: list[str] = []
    doc_only: list[str] = []
    for _cname, cdef in classes:
        if cdef.get("carrier") != "recipe":
            continue
        owners = cdef.get("owner", {}) or {}
        for var in cdef["vars"]:
            checked += 1
            pattern = var_pattern(var)
            owner = owners.get(var)
            owner_text = None
            if owner:
                owner_text = code_by_rel.get(owner)
                if owner_text is not None and pattern.search(owner_text):
                    continue
            elif any(pattern.search(text) for _rel, text in code_chunks):
                # No declared owner: fall back to "somewhere in this recipe's
                # own code", which is the best this check can do.
                continue
            in_docs = any(pattern.search(text) for _rel, text in doc_chunks)
            in_other_code = any(
                pattern.search(text) for rel, text in code_chunks
                if owner is None or rel != owner
            )
            if in_docs:
                # The name survives in documentation. That is not an applier
                # reading it -- it is the README lagging behind the applier
                # that was changed or removed.
                doc_only.append(var)
                where = f" (owner: {owner})" if owner else ""
                gone = (" -- that owner file is missing from the tree"
                        if owner and owner_text is None else "")
                also = (" — and in files that do not own it"
                        if in_other_code else "")
                rep.fail(
                    f"[recipe] {var} appears only in documentation in this recipe's own "
                    "sources — the applier that owned it was changed or removed, and "
                    f"the README still names it{where}{gone}{also}"
                )
                continue
            missing.append(var)
            if owner and owner_text is None:
                rep.fail(
                    f"[recipe] {var} is no longer read by this recipe's own sources — "
                    f"its owner file {owner} is missing from the tree"
                )
            elif owner:
                rep.fail(
                    f"[recipe] {var} is no longer read by this recipe's own sources — "
                    f"its owner {owner} does not read it (another file naming it does "
                    "not count as a reader)"
                )
            else:
                rep.fail(
                    f"[recipe] {var} is no longer read by this recipe's own sources — "
                    "the applier that owned it was changed or removed"
                )
    for var in doc_only:
        emit(quiet, f"    DOC-ONLY VAR {var}")
    for var in missing:
        emit(quiet, f"    MISSING VAR {var}")
    ok = checked - len(missing) - len(doc_only)
    emit(quiet, "  recipe-owned variables: "
                f"{ok}/{checked} still read by the appliers"
                + (f" ({len(doc_only)} documentation-only, "
                   f"{len(missing)} gone)" if doc_only or missing else ""))
    return rep


def check_classification(profile_vars: list[str], classes: list[tuple[str, dict]],
                         manifest: dict, quiet: bool = False) -> Report:
    """Every profile variable must be classified, and every classified one used."""
    rep = Report()
    keys = upstream_keys(manifest)
    owners = ownership_classes(manifest)

    carrier_of: dict[str, list[str]] = {}
    owner_of: dict[str, list[str]] = {}
    for cname, cdef in classes:
        for var in cdef["vars"]:
            if cdef.get("carrier") in keys:
                carrier_of.setdefault(var, []).append(cname)
            else:
                owner_of.setdefault(var, []).append(cname)

    for var, names in sorted(owner_of.items()):
        if len(names) > 1:
            rep.fail(
                f"the manifest gives {var} more than one owner ({', '.join(names)}) — "
                "a variable has exactly one owner"
            )

    unclassified = [v for v in profile_vars if v not in carrier_of and v not in owner_of]
    for var in unclassified:
        rep.fail(
            f"{var} is set by the profile but no manifest class claims it — classify it "
            "before the profile and the upstreams can drift apart"
        )
    stale = [v for v in list(carrier_of) + list(owner_of) if v not in profile_vars]
    for var in stale:
        rep.fail(
            f"the manifest classifies {var}, but the profile no longer sets it — "
            "the manifest is stale"
        )
    if not rep.failures:
        emit(quiet, f"classification: all {len(profile_vars)} profile variables are classified "
              f"({len(carrier_of)} upstream-carried, {len(owner_of)} owned here or reader-level)")
    return rep


def report_reader_library(classes: list[tuple[str, dict]], quiet: bool = False) -> Report:
    """The reader/library class is a WARNING class — never a hard failure."""
    rep = Report()
    for cname, cdef in classes:
        if cname != "reader_library":
            continue
        notes = cdef.get("notes", {})
        emit(quiet, f"WARNING (reader/library-level, not a failure): {len(cdef['vars'])} variables "
              "are carried by no pinned upstream")
        for var in cdef["vars"]:
            note = notes.get(var, "")
            emit(quiet, f"  - {var}: {note}")
            rep.warn(f"reader/library-level variable {var}: {note}")
    return rep


# --------------------------------------------------------------------------
# the run


def run(manifest: dict, profile_vars: list[str], sources: dict[str, Source],
        recipe_root: Path, quiet: bool = False) -> Report:
    classes = list(manifest["classes"].items())
    rep = Report()

    for upstream in manifest["upstreams"]:
        key = upstream["key"]
        if key not in sources:
            rep.fail(
                f"[{key}] no source was given for this upstream — pass a checkout or an "
                "archive, or use --remote"
            )
            continue
        rep.merge(check_upstream(sources[key], upstream, classes, quiet))

    rep.merge(check_recipe(recipe_root, classes,
                           manifest.get("recipe_sources", ["patches", "tools"]), quiet))
    rep.merge(check_classification(profile_vars, classes, manifest, quiet))
    rep.merge(report_reader_library(classes, quiet))
    return rep


def summarise(rep: Report) -> int:
    if rep.failures:
        print()
        print("RESULT: FAIL — the profile and the pinned upstreams no longer agree:")
        for msg in rep.failures:
            print(f"  - {msg}")
        return EXIT_FAIL
    print()
    if rep.warnings:
        print(f"RESULT: PASS with {len(rep.warnings)} reader/library-level warning(s) "
              "(carried by no pinned upstream — verify them for your stack)")
    else:
        print("RESULT: PASS — the profile and the pinned upstreams still agree.")
    return EXIT_PASS


# --------------------------------------------------------------------------
# selftest


def write_synthetic(tree: Path, upstream: dict, values: dict[str, str]) -> None:
    """Write the synthetic variables into every pinned source file of an upstream."""
    body = "# synthetic upstream\n" + "".join(f"{k}={v}\n" for k, v in values.items())
    for rel in upstream.get("search_files", []):
        candidate = tree / rel
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(body)


def selftest(manifest: dict) -> int:
    """Prove the guard can FAIL: rename a variable in a synthetic upstream."""
    print("coupling-guard selftest — offline, synthetic upstream trees")
    ok = True
    classes = list(manifest["classes"].items())
    keys = upstream_keys(manifest)

    with tempfile.TemporaryDirectory(prefix="coupling-guard-selftest-") as tmp:
        root = Path(tmp)

        # 1. build a synthetic tree per upstream that carries its variables
        trees: dict[str, Path] = {}
        synthetic: dict[str, dict[str, str]] = {}
        for upstream in manifest["upstreams"]:
            key = upstream["key"]
            tree = root / key
            tree.mkdir(parents=True, exist_ok=True)
            names: list[str] = []
            for _cname, cdef in classes:
                if cdef.get("carrier") == key:
                    names.extend(cdef["vars"])
            synthetic[key] = {n: "1" for n in names}
            trees[key] = tree
            for rel in upstream.get("search_files", []):
                candidate = tree / rel
                candidate.parent.mkdir(parents=True, exist_ok=True)
            write_synthetic(tree, upstream, synthetic[key])

        # 2. synthetic recipe sources: the recipe-owned variables, read by an applier
        recipe_names: list[str] = []
        for _cname, cdef in classes:
            if cdef.get("carrier") == "recipe":
                recipe_names.extend(cdef["vars"])
        applier = root / "patches" / "apply_synthetic.py"
        applier.parent.mkdir(parents=True, exist_ok=True)
        applier.write_text(
            "# synthetic applier\n"
            + "\n".join(f"os.environ.get('{n}')" for n in recipe_names) + "\n"
        )
        (root / "tools").mkdir(parents=True, exist_ok=True)

        # The synthetic tree has one applier, so the manifest's real per-variable
        # owners are remapped onto it. The owner binding itself is exercised
        # below, against a sibling file that is NOT the owner.
        synth_manifest = copy.deepcopy(manifest)
        synth_manifest["classes"]["recipe_owned"]["owner"] = {
            n: "patches/apply_synthetic.py" for n in recipe_names
        }
        classes = list(synth_manifest["classes"].items())
        keys = upstream_keys(synth_manifest)

        profile_vars: list[str] = []
        for _cname, cdef in classes:
            for var in cdef["vars"]:
                if var not in profile_vars:
                    profile_vars.append(var)

        def sources_now() -> dict[str, Source]:
            return {
                u["key"]: LocalCheckout(u["key"], trees[u["key"]], u.get("search_files", []))
                for u in synth_manifest["upstreams"]
            }

        rep = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
        passed = not rep.failures
        print(f"  {'PASS' if passed else 'FAIL'} selftest: an intact synthetic upstream passes")
        for msg in rep.failures:
            print(f"        {msg}")
        ok = ok and passed

        # 3. rename one variable in the first upstream tree — the guard must FAIL
        victim = "MAX_NUM_SEQS"
        first = synth_manifest["upstreams"][0]
        synthetic[first["key"]][victim + "_RENAMED"] = synthetic[first["key"]].pop(victim)
        write_synthetic(trees[first["key"]], first, synthetic[first["key"]])
        rep2 = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
        detected = any(victim in m and "no longer appears" in m for m in rep2.failures)
        print(f"  {'PASS' if detected else 'FAIL'} selftest: renaming {victim} upstream is detected "
              f"({len(rep2.failures)} failure(s) reported)")
        if detected:
            for msg in rep2.failures:
                if victim in msg:
                    print(f"        {msg}")
        else:
            print("        the guard did not notice the rename — it would be a false pass")
        ok = ok and detected
        synthetic[first["key"]][victim] = "1"
        write_synthetic(trees[first["key"]], first, synthetic[first["key"]])

        # 3. a recipe-owned variable that survives only in DOCUMENTATION is
        #    a FAILURE, not a presence. A README that still names a variable
        #    its applier no longer reads is exactly the drift this guard
        #    exists to catch -- and a search that unions every file's text
        #    would have called it present. This case is the one that proves
        #    the recipe half does not do that.
        if recipe_names:
            readme = root / "patches" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "# synthetic documentation\n"
                + "\n".join(f"# {n} is configured here" for n in recipe_names) + "\n"
            )
            applier.unlink()
            rep_doc = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
            doc_detected = any(
                "appears only in documentation" in m for m in rep_doc.failures
            )
            print(f"  {'PASS' if doc_detected else 'FAIL'} selftest: a recipe-owned variable "
                  f"that survives only in documentation is a failure, not a presence "
                  f"({len(rep_doc.failures)} failure(s) reported)")
            if doc_detected:
                for msg in rep_doc.failures:
                    if "appears only in documentation" in msg:
                        print(f"        {msg}")
            else:
                print("        the guard called a documentation-only variable present — "
                      "a union search would have passed it")
            ok = ok and doc_detected
            # restore the tree for the remaining cases
            applier.parent.mkdir(parents=True, exist_ok=True)
            applier.write_text(
                "# synthetic applier\n"
                + "\n".join(f"os.environ.get('{n}')" for n in recipe_names) + "\n"
            )
            readme.unlink()

            # 3b. a recipe-owned variable whose OWNER file is gone, but whose
            #     name survives in a SIBLING code file, is a FAILURE. The
            #     sibling is not the owner: a probe's docstring or a comment
            #     that happens to mention the knob is not a reader. A union
            #     over every non-doc file would have called it present.
            sibling = root / "tools" / "sibling_reader.py"
            sibling.write_text(
                "# synthetic sibling: mentions the knobs but does not own them\n"
                + "\n".join(f"# {n}" for n in recipe_names) + "\n"
            )
            applier.unlink()
            rep_sib = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
            sib_detected = any(
                "is no longer read by this recipe's own sources" in m
                for m in rep_sib.failures
            )
            print(f"  {'PASS' if sib_detected else 'FAIL'} selftest: a recipe-owned variable "
                  f"named only in a sibling file is a failure — the owner binding holds "
                  f"({len(rep_sib.failures)} failure(s) reported)")
            if sib_detected:
                for msg in rep_sib.failures:
                    if "is no longer read by this recipe's own sources" in msg:
                        print(f"        {msg}")
            else:
                print("        the guard called a sibling's mention a reader — "
                      "a union search would have passed it")
            ok = ok and sib_detected
            sibling.unlink()
            applier.parent.mkdir(parents=True, exist_ok=True)
            applier.write_text(
                "# synthetic applier\n"
                + "\n".join(f"os.environ.get('{n}')" for n in recipe_names) + "\n"
            )

        # 4. a reader/library-level variable is a warning class, never a failure:
        #    it must not change the verdict whether it is present or absent
        rl_vars = synth_manifest["classes"].get("reader_library", {}).get("vars", [])
        if rl_vars:
            rl_var = rl_vars[0]
            synthetic[first["key"]][rl_var] = "1"
            write_synthetic(trees[first["key"]], first, synthetic[first["key"]])
            rep3 = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
            present_ok = not rep3.failures
            synthetic[first["key"]].pop(rl_var, None)
            write_synthetic(trees[first["key"]], first, synthetic[first["key"]])
            rep4 = run(synth_manifest, profile_vars, sources_now(), root, quiet=True)
            absent_ok = not rep4.failures
            clean = present_ok and absent_ok
            print(f"  {'PASS' if clean else 'FAIL'} selftest: a reader/library-level variable "
                  "is a warning, not a failure (present and absent)")
            for msg in rep3.failures + rep4.failures:
                print(f"        {msg}")
            ok = ok and clean

        # 5. an unreachable network must be INCONCLUSIVE, never a pass
        remote = RemoteSource(
            "compose", "example/does-not-matter", "0" * 40,
            raw_base="http://127.0.0.1:9", tar_base="http://127.0.0.1:9", timeout=2,
        )
        status, _rev, rnote = remote.revision()
        net_ok = status == "unreachable"
        print(f"  {'PASS' if net_ok else 'FAIL'} selftest: an unreachable upstream is reported "
              f"as unreachable, never as a pass ({rnote})")
        ok = ok and net_ok

        # 6. the reader/library class is never a hard failure
        reader_class = manifest["classes"].get("reader_library", {})
        reader_ok = reader_class.get("carrier") is None
        print(f"  {'PASS' if reader_ok else 'FAIL'} selftest: the reader/library class carries no "
              "upstream, so it can only warn")
        ok = ok and reader_ok

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# cli


def build_parser(manifest: dict) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="the coupling manifest (default: config/coupling-manifest.json)")
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE),
                    help="the profile whose variables are checked (default: config/cluster.env)")
    ap.add_argument("--remote", action="store_true",
                    help="check the pinned revisions on the live upstream (needs the network)")
    ap.add_argument("--timeout", type=int, default=30, help="network timeout in seconds")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the guard can fail (offline, synthetic trees)")
    for upstream in manifest["upstreams"]:
        key = upstream["key"]
        ap.add_argument(f"--{key}-dir", default=None,
                        help=f"a pinned checkout of the {key} upstream")
        ap.add_argument(f"--{key}-archive", default=None,
                        help=f"a tar archive of the pinned {key} upstream")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # the manifest names the upstream keys, so load it before parsing
    manifest_path = DEFAULT_MANIFEST
    profile_path = DEFAULT_PROFILE
    for i, arg in enumerate(argv):
        if arg == "--manifest" and i + 1 < len(argv):
            manifest_path = Path(argv[i + 1])
        elif arg.startswith("--manifest="):
            manifest_path = Path(arg.split("=", 1)[1])
        elif arg == "--profile" and i + 1 < len(argv):
            profile_path = Path(argv[i + 1])
        elif arg.startswith("--profile="):
            profile_path = Path(arg.split("=", 1)[1])

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        print(f"cannot read the coupling manifest {manifest_path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        profile_vars = parse_profile(profile_path)
    except OSError as exc:
        print(f"cannot read the profile {profile_path}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    ap = build_parser(manifest)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error, and 2 is this tool's
        # INCONCLUSIVE code. A malformed command line must not be
        # indistinguishable from "the upstream could not be reached".
        if exc.code not in (0, None):
            return EXIT_USAGE
        return EXIT_PASS

    if args.selftest:
        return selftest(manifest)

    print(f"coupling-guard — profile: {Path(args.profile).name} ({len(profile_vars)} variables)")
    print(f"manifest: {Path(args.manifest).name} (version {manifest.get('manifest_version', '?')})")
    print()

    sources: dict[str, Source] = {}
    usage_errors: list[str] = []
    for upstream in manifest["upstreams"]:
        key = upstream["key"]
        directory = getattr(args, f"{key}_dir", None)
        archive = getattr(args, f"{key}_archive", None)
        if directory and archive:
            usage_errors.append(f"pass either --{key}-dir or --{key}-archive, not both")
            continue
        if directory:
            path = Path(directory)
            if not path.is_dir():
                usage_errors.append(f"--{key}-dir {path} is not a directory")
                continue
            sources[key] = LocalCheckout(key, path, upstream.get("search_files", []))
        elif archive:
            path = Path(archive)
            if not path.is_file():
                usage_errors.append(f"--{key}-archive {path} is not a file")
                continue
            sources[key] = LocalArchive(key, path)
        elif args.remote:
            sources[key] = RemoteSource(key, upstream["repo"], upstream["rev"], timeout=args.timeout)

    if usage_errors:
        for msg in usage_errors:
            print(f"usage error: {msg}", file=sys.stderr)
        return EXIT_USAGE

    if not sources:
        print("usage error: give a pinned checkout or archive for each upstream, "
              "or pass --remote", file=sys.stderr)
        return EXIT_USAGE

    rep = run(manifest, profile_vars, sources, RECIPE_ROOT)

    if rep.inconclusive:
        print()
        print("WARNING: the upstream could not be reached — the coupling check is "
              "INCONCLUSIVE, not a pass.")
        for msg in rep.inconclusive:
            print(f"  - {msg}")
        print("RESULT: INCONCLUSIVE — the check could not complete.")
        return EXIT_INCONCLUSIVE

    return summarise(rep)


if __name__ == "__main__":
    sys.exit(main())
