//! Pure-Rust LMDB-compatible persistence implementation for GFlowNet-Prover.

use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Database error: {0}")]
    Custom(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub mod types {
    use std::marker::PhantomData;

    pub struct Bytes;
    pub struct ByteSlice;
    pub struct Str;
    pub struct SerdeBincode<T>(pub PhantomData<T>);
    pub struct SerdeJson<T>(pub PhantomData<T>);
}

#[derive(Debug, Clone, Default)]
pub struct EnvOpenOptions {
    max_dbs: Option<u32>,
    map_size: Option<usize>,
}

impl EnvOpenOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_dbs(&mut self, max: u32) -> &mut Self {
        self.max_dbs = Some(max);
        self
    }

    pub fn map_size(&mut self, size: usize) -> &mut Self {
        self.map_size = Some(size);
        self
    }

    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<Env> {
        let path_buf = path.as_ref().to_path_buf();
        fs::create_dir_all(&path_buf)?;

        let store_file = path_buf.join("data.bin");
        let mut data = BTreeMap::new();
        if store_file.exists() {
            if let Ok(mut f) = File::open(&store_file) {
                let mut buf = Vec::new();
                if f.read_to_end(&mut buf).is_ok() && !buf.is_empty() {
                    if let Ok(deser) = serde_json::from_slice::<BTreeMap<String, BTreeMap<String, String>>>(&buf) {
                        for (db_name, kvs) in deser {
                            let mut inner_map = BTreeMap::new();
                            for (hex_k, hex_v) in kvs {
                                if let (Ok(k), Ok(v)) = (hex::decode(hex_k), hex::decode(hex_v)) {
                                    inner_map.insert(k, v);
                                }
                            }
                            data.insert(db_name, inner_map);
                        }
                    }
                }
            }
        }

        Ok(Env {
            path: path_buf,
            databases: Arc::new(RwLock::new(data)),
        })
    }
}

#[derive(Clone)]
pub struct Env {
    path: PathBuf,
    databases: Arc<RwLock<BTreeMap<String, BTreeMap<Vec<u8>, Vec<u8>>>>>,
}

impl Env {
    pub fn read_txn(&self) -> Result<RoTxn> {
        let guard = self.databases.read();
        Ok(RoTxn {
            snapshot: guard.clone(),
        })
    }

    pub fn write_txn(&self) -> Result<RwTxn> {
        Ok(RwTxn {
            env: self.clone(),
            pending: BTreeMap::new(),
        })
    }

    pub fn create_database<KC, VC>(&self, _txn: &mut RwTxn, name: Option<&str>) -> Result<Database<KC, VC>> {
        let db_name = name.unwrap_or("default").to_string();
        let mut dbs = self.databases.write();
        dbs.entry(db_name.clone()).or_default();
        Ok(Database {
            name: db_name,
            _phantom: PhantomData,
        })
    }

    pub fn open_database<KC, VC>(&self, _txn: &RoTxn, name: Option<&str>) -> Result<Option<Database<KC, VC>>> {
        let db_name = name.unwrap_or("default").to_string();
        let dbs = self.databases.read();
        if dbs.contains_key(&db_name) {
            Ok(Some(Database {
                name: db_name,
                _phantom: PhantomData,
            }))
        } else {
            Ok(None)
        }
    }
}

pub struct RoTxn {
    snapshot: BTreeMap<String, BTreeMap<Vec<u8>, Vec<u8>>>,
}

pub struct RwTxn {
    env: Env,
    pending: BTreeMap<String, BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl RwTxn {
    pub fn commit(self) -> Result<()> {
        let mut dbs = self.env.databases.write();
        for (db_name, entries) in self.pending {
            let db = dbs.entry(db_name).or_default();
            for (k, v) in entries {
                db.insert(k, v);
            }
        }

        // Persist to disk
        let store_file = self.env.path.join("data.bin");
        let mut serialized_dbs = BTreeMap::new();
        for (db_name, inner) in dbs.iter() {
            let mut hex_map = BTreeMap::new();
            for (k, v) in inner {
                hex_map.insert(hex::encode(k), hex::encode(v));
            }
            serialized_dbs.insert(db_name.clone(), hex_map);
        }

        if let Ok(bytes) = serde_json::to_vec(&serialized_dbs) {
            let tmp_file = self.env.path.join("data.bin.tmp");
            if let Ok(mut f) = File::create(&tmp_file) {
                let _ = f.write_all(&bytes);
                let _ = f.sync_all();
                let _ = fs::rename(tmp_file, store_file);
            }
        }

        Ok(())
    }

    pub fn abort(self) {}
}

#[derive(Clone)]
pub struct Database<KC, VC> {
    name: String,
    _phantom: PhantomData<(KC, VC)>,
}

impl<KC, VC> Database<KC, VC> {
    pub fn get<'a>(&self, txn: &'a RoTxn, key: &[u8]) -> Result<Option<&'a [u8]>> {
        if let Some(db) = txn.snapshot.get(&self.name) {
            if let Some(val) = db.get(key) {
                return Ok(Some(val.as_slice()));
            }
        }
        Ok(None)
    }

    pub fn put(&self, txn: &mut RwTxn, key: &[u8], val: &[u8]) -> Result<()> {
        let db_entries = txn.pending.entry(self.name.clone()).or_default();
        db_entries.insert(key.to_vec(), val.to_vec());
        Ok(())
    }
}

// Simple hex module for zero-dependency hex encoding/decoding in heed stub
mod hex {
    pub fn encode(data: &[u8]) -> String {
        let mut s = String::with_capacity(data.len() * 2);
        for &b in data {
            use std::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    pub fn decode(s: String) -> Result<Vec<u8>, ()> {
        if s.len() % 2 != 0 {
            return Err(());
        }
        let mut bytes = Vec::with_capacity(s.len() / 2);
        for i in (0..s.len()).step_by(2) {
            let byte = u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ())?;
            bytes.push(byte);
        }
        Ok(bytes)
    }
}
