"""Small rebuildable index; immutable run artifacts remain the source of truth."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tsmcp.cleaning.common import digest, read, write


class Store:
    def __init__(self, repo, state_dir=None):
        self.repo = Path(repo).resolve()
        self.root = Path(state_dir or self.repo / ".signalmind-workbench").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "index.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, root TEXT UNIQUE, info TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, info TEXT)")
        for path in sorted((self.repo / "eval/runs").glob("vital-cleaning-v1-20261112*")):
            if (path / "dataset_manifest.json").exists():
                self.add_run(path, historical=True)
        for path in (self.root / "runs").glob("*/run.json"):
            self.add_run(path.parent)
        for path in (self.root / "datasets").glob("*.json"):
            self.save_dataset(read(path))
        vital = self.repo.parent / "bp2rr_new/dataset/Vital/PulseDB_Vital"
        inventory = self.repo.parent / "bp2rr_new/preprocess/data/v1_20260920/inventory.json"
        if vital.is_dir() and inventory.is_file() and not any(d["path"] == str(vital) for d in self.datasets()):
            self.save_dataset(
                {"name": "PulseDB Vital", "format": "vital", "path": str(vital), "inventory": str(inventory)}
            )

    def connect(self):
        db = sqlite3.connect(self.db, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def add_run(self, path, historical=False):
        path = Path(path).resolve()
        rid = path.name
        info = {"id": rid, "historical": historical}
        if (path / "run.json").exists():
            info.update(read(path / "run.json"))
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)", (rid, str(path), json.dumps(info)))
        return rid

    def run_path(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT root FROM runs WHERE id=?", (rid,)).fetchone()
        if not row:
            raise KeyError("Unknown run")
        return Path(row["root"])

    def runs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY id DESC")]

    def save_dataset(self, item):
        item = dict(item)
        item.setdefault("id", "d_" + digest([item["path"], item["format"]])[:16])
        write(self.root / "datasets" / (item["id"] + ".json"), item)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO datasets VALUES (?, ?)", (item["id"], json.dumps(item)))
        return item

    def datasets(self):
        with self.connect() as db:
            return [json.loads(row["info"]) for row in db.execute("SELECT info FROM datasets ORDER BY id")]

    def dataset(self, did):
        with self.connect() as db:
            row = db.execute("SELECT info FROM datasets WHERE id=?", (did,)).fetchone()
        if not row:
            raise KeyError("Unknown dataset")
        return json.loads(row["info"])
