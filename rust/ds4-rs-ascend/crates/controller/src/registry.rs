// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The local kernel-set registry, SQLite-backed.
//!
//! A "kernel set" is a portable CONFIG (integer tuning knobs + SoC class) plus optional
//! toolchain-specific ELF attachments. Sets come from three provenances — `bundled` (shipped,
//! immovable floor), `published` (downloaded, signed, re-fetchable), `profiled` (self-tuned, fully
//! owned) — and the protection rules follow from that: only `profiled` sets can be destroyed;
//! `published` bytes can be evicted (re-fetchable); `bundled` is immovable.
//!
//! The set id is content-addressed (`sha256(canonical config)[..12]`), so integrity, dedup and
//! idempotent sharing all fall out of the id itself.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: &str = "1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Shipped with the binary. The floor: never deletable, never evictable, no re-fetch source.
    Bundled,
    /// Downloaded from published.ai, Ed25519-signed. Not `rm`-deletable, but evictable (re-fetchable).
    Publisher,
    /// Self-profiled on this box. Fully owned — the only kind `rm` destroys.
    Profiled,
}

impl Provenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::Bundled => "bundled",
            Provenance::Publisher => "published",
            Provenance::Profiled => "profiled",
        }
    }
    pub fn parse(s: &str) -> Result<Provenance> {
        Ok(match s {
            "bundled" => Provenance::Bundled,
            "published" => Provenance::Publisher,
            "profiled" => Provenance::Profiled,
            o => bail!("unknown provenance {o:?}"),
        })
    }
    /// `rm` destroys only profiled sets .
    pub fn deletable(self) -> bool {
        matches!(self, Provenance::Profiled)
    }
    /// `evict` frees local bytes only for published sets (re-fetchable; decision 10).
    pub fn evictable(self) -> bool {
        matches!(self, Provenance::Publisher)
    }
}

/// A row of the kernel-set table, as read back.
///
/// `created_utc` and `source_url` are not read by the command surface yet. They
/// are columns the registry WRITES, so dropping them from this struct would
/// make the read-back a lossy view of its own table -- and the next reader would
/// have to go to the schema to find out what is actually stored.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KernelSet {
    pub id: String,
    pub provenance: Provenance,
    pub label: String,
    pub created_utc: i64,
    pub soc_arch: String,
    pub config_json: String,
    pub source_url: Option<String>,
    pub verified: bool,
    pub active: bool,
    pub bytes_present: bool,
}

/// `sha256` hex of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let h = sha2::Sha256::digest(bytes);
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// Canonicalize a config's JSON before hashing, so semantically identical configs get the same
/// id regardless of key order or whitespace. Minimal canonicalization: the config is emitted by
/// us (profile.rs) in a fixed key order already, so this trims and collapses whitespace runs —
/// enough for our own output while staying dependency-free (no JSON parser pulled in).
pub fn canonical_config(config_json: &str) -> String {
    let mut out = String::with_capacity(config_json.len());
    let mut in_str = false;
    let mut prev_ws = false;
    for c in config_json.chars() {
        if c == '"' {
            in_str = !in_str;
        }
        if !in_str && c.is_whitespace() {
            if !prev_ws {
                // drop insignificant whitespace entirely rather than collapse — key order is
                // fixed by the emitter, so the token stream is stable without it.
            }
            prev_ws = true;
            continue;
        }
        prev_ws = false;
        out.push(c);
    }
    out
}

/// The content-addressed id for a config: `sha256(canonical)[..12]`.
pub fn config_id(config_json: &str) -> String {
    sha256_hex(canonical_config(config_json).as_bytes())[..12].to_string()
}

pub struct Registry {
    conn: Connection,
    root: PathBuf,
}

impl Registry {
    /// Default DB location: `~/.local/share/ds4/kernels.db`, sets under `.../kernels/<id>/`.
    pub fn default_path() -> Result<PathBuf> {
        let home = std::env::var("HOME").context("registry: HOME not set")?;
        Ok(PathBuf::from(home).join(".local/share/ds4"))
    }

    pub fn open_default() -> Result<Registry> {
        Self::open(&Self::default_path()?)
    }

    /// Open (creating if needed) the registry rooted at `dir`.
    pub fn open(dir: &Path) -> Result<Registry> {
        std::fs::create_dir_all(dir).with_context(|| format!("registry dir {}", dir.display()))?;
        let conn = Connection::open(dir.join("kernels.db")).context("open kernels.db")?;
        let reg = Registry { conn, root: dir.to_path_buf() };
        reg.init()?;
        Ok(reg)
    }

    fn init(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS kernel_set(
                id TEXT PRIMARY KEY, provenance TEXT NOT NULL, label TEXT NOT NULL,
                created_utc INTEGER NOT NULL, soc_arch TEXT NOT NULL, config_json TEXT NOT NULL,
                source_url TEXT, verified INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                bytes_present INTEGER NOT NULL DEFAULT 1);
             CREATE TABLE IF NOT EXISTS fingerprint(
                fp TEXT PRIMARY KEY, soc_arch TEXT, soc_name TEXT,
                cann_version TEXT, ptoas_version TEXT, note TEXT);
             CREATE TABLE IF NOT EXISTS elf(
                set_id TEXT NOT NULL, fp TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
                PRIMARY KEY(set_id, fp));
             CREATE TABLE IF NOT EXISTS measurement(
                set_id TEXT NOT NULL, fp TEXT NOT NULL, tok_s REAL NOT NULL,
                measured_utc INTEGER NOT NULL, PRIMARY KEY(set_id, fp));
             CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);",
        )?;
        // schema_version + a stable random install-id (decision 7 — never the email).
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION],
        )?;
        if self.meta("install_id")?.is_none() {
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('install_id', ?1)",
                params![new_install_id()],
            )?;
        }
        Ok(())
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    /// The random install-id . Present after `init`.
    pub fn install_id(&self) -> Result<String> {
        self.meta("install_id")?
            .context("registry: install_id missing")
    }

    /// Directory holding a set's bytes (config file + any ELFs).
    pub fn set_dir(&self, id: &str) -> PathBuf {
        self.root.join("kernels").join(id)
    }

    /// Register a set. The id is derived from the config, so re-registering identical config is
    /// idempotent ; provenance/label/verified are updated on conflict.
    pub fn register(
        &self,
        provenance: Provenance,
        label: &str,
        soc_arch: &str,
        config_json: &str,
        source_url: Option<&str>,
        verified: bool,
    ) -> Result<String> {
        let id = config_id(config_json);
        let dir = self.set_dir(&id);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("config.json"), config_json)?;
        self.conn.execute(
            "INSERT INTO kernel_set(id, provenance, label, created_utc, soc_arch, config_json,
                    source_url, verified, active, bytes_present)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,0,1)
             ON CONFLICT(id) DO UPDATE SET label=excluded.label, verified=excluded.verified,
                    source_url=excluded.source_url, bytes_present=1",
            params![
                id,
                provenance.as_str(),
                label,
                now_secs(),
                soc_arch,
                config_json,
                source_url,
                verified as i64,
            ],
        )?;
        Ok(id)
    }

    fn row_to_set(r: &rusqlite::Row) -> rusqlite::Result<KernelSet> {
        Ok(KernelSet {
            id: r.get(0)?,
            provenance: Provenance::parse(&r.get::<_, String>(1)?)
                .map_err(|e| rusqlite::Error::InvalidColumnType(1, e.to_string(), rusqlite::types::Type::Text))?,
            label: r.get(2)?,
            created_utc: r.get(3)?,
            soc_arch: r.get(4)?,
            config_json: r.get(5)?,
            source_url: r.get(6)?,
            verified: r.get::<_, i64>(7)? != 0,
            active: r.get::<_, i64>(8)? != 0,
            bytes_present: r.get::<_, i64>(9)? != 0,
        })
    }

    const COLS: &'static str =
        "id, provenance, label, created_utc, soc_arch, config_json, source_url, verified, active, bytes_present";

    pub fn get(&self, id: &str) -> Result<Option<KernelSet>> {
        let sql = format!("SELECT {} FROM kernel_set WHERE id = ?1", Self::COLS);
        Ok(self
            .conn
            .query_row(&sql, params![id], Self::row_to_set)
            .optional()?)
    }

    pub fn list(&self) -> Result<Vec<KernelSet>> {
        let sql = format!(
            "SELECT {} FROM kernel_set ORDER BY active DESC, created_utc DESC",
            Self::COLS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::row_to_set)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    // Not yet called from the command surface: this is the registry API for
    // selecting which kernel set a deploy uses. Kept rather than deleted because the schema it writes is
    // already on disk, and a reader should see the write path that produced it.
    #[allow(dead_code)]
    pub fn active(&self) -> Result<Option<KernelSet>> {
        let sql = format!("SELECT {} FROM kernel_set WHERE active = 1", Self::COLS);
        Ok(self
            .conn
            .query_row(&sql, [], Self::row_to_set)
            .optional()?)
    }

    /// Make `id` the sticky default . Verifies integrity first : a set
    /// that fails its content hash is refused rather than activated.
    pub fn set_active(&self, id: &str) -> Result<()> {
        let set = self.get(id)?.with_context(|| format!("no set {id}"))?;
        self.verify(&set)?;
        self.conn.execute("UPDATE kernel_set SET active = 0", [])?;
        self.conn
            .execute("UPDATE kernel_set SET active = 1 WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Destroy a set : profiled only. Removes DB rows and the on-disk bytes.
    pub fn rm(&self, id: &str) -> Result<()> {
        let set = self.get(id)?.with_context(|| format!("no set {id}"))?;
        if !set.provenance.deletable() {
            bail!(
                "refusing to delete {} set {id}: only self-profiled sets can be removed. \
                 {}",
                set.provenance.as_str(),
                match set.provenance {
                    Provenance::Publisher => "Use `kernels evict` to free its bytes (it re-downloads).",
                    Provenance::Bundled => "Bundled sets are the install floor and are immovable.",
                    Provenance::Profiled => unreachable!(),
                }
            );
        }
        if set.active {
            bail!("{id} is the active set; `kernels use <other>` before removing it");
        }
        self.conn.execute("DELETE FROM elf WHERE set_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM measurement WHERE set_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM kernel_set WHERE id = ?1", params![id])?;
        let _ = std::fs::remove_dir_all(self.set_dir(id));
        Ok(())
    }

    /// Free a published set's local bytes while keeping its verified, re-fetchable index stub
    /// : `bytes_present -> 0`, ELF attachments dropped, config kept in the DB row.
    pub fn evict(&self, id: &str) -> Result<()> {
        let set = self.get(id)?.with_context(|| format!("no set {id}"))?;
        if !set.provenance.evictable() {
            bail!(
                "cannot evict a {} set: only published downloads are re-fetchable and so evictable",
                set.provenance.as_str()
            );
        }
        if set.active {
            bail!("{id} is the active set; `kernels use <other>` before evicting it");
        }
        self.conn.execute("DELETE FROM elf WHERE set_id = ?1", params![id])?;
        self.conn
            .execute("UPDATE kernel_set SET bytes_present = 0 WHERE id = ?1", params![id])?;
        // Keep config.json (tiny); drop only the ELF bytes on disk.
        let dir = self.set_dir(id);
        if dir.exists() {
            for entry in std::fs::read_dir(&dir)?.flatten() {
                if entry.path().extension().map(|e| e == "o").unwrap_or(false) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }

    /// Verify a set's integrity : the stored config must still hash to its id, and
    /// every present ELF must match its pinned sha256.
    pub fn verify(&self, set: &KernelSet) -> Result<()> {
        let recomputed = config_id(&set.config_json);
        if recomputed != set.id {
            bail!(
                "set {} FAILED verification: config hashes to {recomputed}, not its id",
                set.id
            );
        }
        let mut stmt = self
            .conn
            .prepare("SELECT path, sha256 FROM elf WHERE set_id = ?1")?;
        let elfs = stmt
            .query_map(params![set.id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (path, want) in elfs {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("verify {}: reading ELF {path}", set.id))?;
            let got = sha256_hex(&bytes);
            if got != want {
                bail!("set {} ELF {path} FAILED verification: {got} != pinned {want}", set.id);
            }
        }
        Ok(())
    }

    /// Attach a compiled ELF to a set, keyed by the full fingerprint (decision 1/2), pinning its
    /// hash .
    // Not yet called from the command surface: this is the registry API for
    // recording a compiled kernel against its set. Kept rather than deleted because the schema it writes is
    // already on disk, and a reader should see the write path that produced it.
    #[allow(dead_code)]
    pub fn attach_elf(&self, set_id: &str, fp: &str, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("reading ELF {}", path.display()))?;
        let sha = sha256_hex(&bytes);
        self.conn.execute(
            "INSERT INTO elf(set_id, fp, path, sha256) VALUES (?1,?2,?3,?4)
             ON CONFLICT(set_id, fp) DO UPDATE SET path=excluded.path, sha256=excluded.sha256",
            params![set_id, fp, path.to_string_lossy(), sha],
        )?;
        Ok(())
    }

    /// Record a fingerprint row .
    pub fn record_fingerprint(
        &self,
        fp: &str,
        soc_arch: &str,
        soc_name: &str,
        cann: &str,
        ptoas: &str,
        note: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO fingerprint(fp, soc_arch, soc_name, cann_version, ptoas_version, note)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![fp, soc_arch, soc_name, cann, ptoas, note],
        )?;
        Ok(())
    }

    /// Record a local measurement  — the authoritative rank key.
    // Not yet called from the command surface: this is the registry API for
    // recording what the tuning step measured. Kept rather than deleted because the schema it writes is
    // already on disk, and a reader should see the write path that produced it.
    #[allow(dead_code)]
    pub fn record_measurement(&self, set_id: &str, fp: &str, tok_s: f64) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO measurement(set_id, fp, tok_s, measured_utc)
             VALUES (?1,?2,?3,?4)",
            params![set_id, fp, tok_s, now_secs()],
        )?;
        Ok(())
    }

    /// Best local measurement for a set on any fingerprint (for `list` display).
    pub fn best_measurement(&self, set_id: &str) -> Result<Option<f64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT MAX(tok_s) FROM measurement WHERE set_id = ?1",
                params![set_id],
                |r| r.get::<_, Option<f64>>(0),
            )
            .optional()?
            .flatten())
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// A random install-id: 128 bits of process/time entropy, hex. Not derived from hardware — it is
/// a random tag, not a machine fingerprint .
fn new_install_id() -> String {
    // No rng dependency: mix time, pid, and a stack address through sha256. Uniqueness, not
    // cryptographic unpredictability, is what an install tag needs.
    let mut seed = Vec::new();
    seed.extend_from_slice(&now_secs().to_le_bytes());
    seed.extend_from_slice(
        &std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    seed.extend_from_slice(&std::process::id().to_le_bytes());
    let local = 0u8;
    seed.extend_from_slice(&(&local as *const u8 as usize).to_le_bytes());
    sha256_hex(&seed)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ds4reg-{}-{}",
            std::process::id(),
            now_secs()
        ));
        // Uniquify further so parallel tests don't collide.
        d.join(format!("{:p}", &d as *const _))
    }

    const CFG_A: &str = r#"{ "knobs": { "matvec_block_dim": 48, "attn_s_block": 8 } }"#;
    const CFG_B: &str = r#"{"knobs":{"matvec_block_dim":40,"attn_s_block":16}}"#;

    #[test]
    fn id_is_content_addressed_and_whitespace_stable() {
        // Same config, different whitespace -> same id .
        let spaced = r#"{ "knobs" : { "matvec_block_dim" : 48 } }"#;
        let tight = r#"{"knobs":{"matvec_block_dim":48}}"#;
        assert_eq!(config_id(spaced), config_id(tight));
        // Different config -> different id.
        assert_ne!(config_id(CFG_A), config_id(CFG_B));
    }

    #[test]
    fn register_is_idempotent_and_install_id_is_stable() {
        let reg = Registry::open(&tmp()).unwrap();
        let id1 = reg.register(Provenance::Profiled, "a", "a3", CFG_A, None, true).unwrap();
        let id2 = reg.register(Provenance::Profiled, "a-again", "a3", CFG_A, None, true).unwrap();
        assert_eq!(id1, id2, "same config -> same id");
        assert_eq!(reg.list().unwrap().len(), 1, "no duplicate row");
        // install-id is present and stable across reopen.
        let iid = reg.install_id().unwrap();
        assert_eq!(iid.len(), 16);
    }

    #[test]
    fn protection_rules_hold() {
        let reg = Registry::open(&tmp()).unwrap();
        let prof = reg.register(Provenance::Profiled, "p", "a3", CFG_A, None, true).unwrap();
        let published = reg.register(Provenance::Publisher, "v1", "a3", CFG_B, Some("https://example.invalid/x"), true).unwrap();
        let bundled = reg.register(Provenance::Bundled, "ship", "a3", r#"{"k":1}"#, None, true).unwrap();

        // profiled: deletable; published/bundled: not.
        assert!(reg.rm(&published).is_err(), "published must not be rm-able");
        assert!(reg.rm(&bundled).is_err(), "bundled must not be rm-able");
        assert!(reg.rm(&prof).is_ok(), "profiled must be rm-able");

        // evict: published only.
        assert!(reg.evict(&bundled).is_err(), "bundled not evictable");
        assert!(reg.evict(&published).is_ok(), "published evictable");
        let ev = reg.get(&published).unwrap().unwrap();
        assert!(!ev.bytes_present, "evicted set keeps its stub with bytes_present=0");
        assert!(reg.get(&published).unwrap().is_some(), "evict must NOT delete the row");
    }

    #[test]
    fn set_active_verifies_and_is_exclusive() {
        let reg = Registry::open(&tmp()).unwrap();
        let a = reg.register(Provenance::Publisher, "va", "a3", CFG_A, Some("u"), true).unwrap();
        let b = reg.register(Provenance::Publisher, "vb", "a3", CFG_B, Some("u"), true).unwrap();
        reg.set_active(&a).unwrap();
        assert_eq!(reg.active().unwrap().unwrap().id, a);
        reg.set_active(&b).unwrap();
        assert_eq!(reg.active().unwrap().unwrap().id, b, "active is exclusive");
        // Exactly one active.
        assert_eq!(reg.list().unwrap().iter().filter(|s| s.active).count(), 1);
    }

    #[test]
    fn verify_catches_config_tampering() {
        let reg = Registry::open(&tmp()).unwrap();
        let id = reg.register(Provenance::Publisher, "v", "a3", CFG_A, Some("u"), true).unwrap();
        // Tamper the stored config_json directly, keeping the old id.
        reg.conn
            .execute("UPDATE kernel_set SET config_json = ?1 WHERE id = ?2", params![CFG_B, id])
            .unwrap();
        let tampered = reg.get(&id).unwrap().unwrap();
        assert!(reg.verify(&tampered).is_err(), "config no longer hashes to its id");
        assert!(reg.set_active(&id).is_err(), "a tampered set must not activate");
    }

    #[test]
    fn measurement_is_the_rank_key() {
        let reg = Registry::open(&tmp()).unwrap();
        let id = reg.register(Provenance::Profiled, "p", "a3", CFG_A, None, true).unwrap();
        assert!(reg.best_measurement(&id).unwrap().is_none());
        reg.record_measurement(&id, "fp1", 5.0).unwrap();
        reg.record_measurement(&id, "fp2", 6.7).unwrap();
        assert_eq!(reg.best_measurement(&id).unwrap(), Some(6.7));
    }
}
