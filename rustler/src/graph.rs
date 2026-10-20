use crate::common::{ArchivedAdj, ArchivedNode, ArchivedOffsets, Offsets};
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use rand::seq::index;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rkyv::rancor::Error;
use std::collections::HashSet;
use std::env::var;
use std::fs;
use std::io::{BufReader, Read};

struct TableRange {
    name: String,
    offset: i32,
    count: i32,
}

#[pyclass]
pub struct GraphInspector {
    mmap: Mmap,
    offsets: Vec<i64>,
    p2f_adj_mmap: Mmap,
    tables: Vec<TableRange>,
    f2p_adj: Vec<Vec<i32>>,
}

#[pymethods]
impl GraphInspector {
    #[new]
    fn new(db_name: &str) -> PyResult<Self> {
        let pre_root = var("NANORT_PRE_ROOT")
            .unwrap_or_else(|_| format!("{}/.cache/modded-nanort/pre", var("HOME").unwrap()));
        let pre_path = format!("{}/{}", pre_root, db_name);

        let nodes_file = fs::File::open(format!("{}/nodes.rkyv", pre_path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&nodes_file) }
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let offsets_file = fs::File::open(format!("{}/offsets.rkyv", pre_path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut bytes = Vec::new();
        BufReader::new(offsets_file).read_to_end(&mut bytes).unwrap();
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let offsets = rkyv::deserialize::<Offsets, Error>(archived)
            .unwrap()
            .offsets;

        let p2f_file = fs::File::open(format!("{}/p2f_adj.rkyv", pre_path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let p2f_adj_mmap = unsafe { Mmap::map(&p2f_file) }
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let content = fs::read_to_string(format!("{}/table_info.json", pre_path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let table_json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| PyIOError::new_err(e.to_string()))?;

        let mut tables: Vec<TableRange> = table_json
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, val)| TableRange {
                name: key.clone(),
                offset: val["node_idx_offset"].as_i64().unwrap() as i32,
                count: val["num_nodes"].as_i64().unwrap() as i32,
            })
            .collect();
        tables.sort_by_key(|t| t.offset);

        let n = offsets.len().saturating_sub(1);
        let mut f2p_adj = vec![vec![]; n];
        {
            let p2f = unsafe { rkyv::access_unchecked::<ArchivedAdj>(&p2f_adj_mmap[..]) };
            for (parent_idx, children) in p2f.adj.iter().enumerate() {
                for edge in children.iter() {
                    let child_idx: i32 = edge.node_idx.into();
                    if (child_idx as usize) < n {
                        f2p_adj[child_idx as usize].push(parent_idx as i32);
                    }
                }
            }
        }

        Ok(GraphInspector { mmap, offsets, p2f_adj_mmap, tables, f2p_adj })
    }

    fn table_names(&self) -> Vec<String> {
        self.tables.iter().map(|t| t.name.clone()).collect()
    }

    fn num_nodes(&self, table: &str) -> usize {
        self.tables
            .iter()
            .find(|t| t.name == table)
            .map_or(0, |t| t.count as usize)
    }

    // Returns (child_table, parent_table) pairs inferred by sampling f2p_edges.
    fn fk_relationships(&self) -> Vec<(String, String)> {
        let mut pairs: HashSet<(String, String)> = HashSet::new();
        let max_offset = (self.offsets.len() as i32).saturating_sub(1);
        for table in &self.tables {
            let sample_end = (table.offset + table.count.min(500)).min(max_offset);
            for idx in table.offset..sample_end {
                let node = self.get_node(idx);
                for edge in node.f2p_edges.iter() {
                    let parent_idx: i32 = edge.node_idx.into();
                    if let Some(dst) = self.table_for_node(parent_idx) {
                        pairs.insert((table.name.clone(), dst));
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }

    fn parent_to_children(&self, parent_node_idx: u32) -> Vec<u32> {
        let p2f = unsafe { rkyv::access_unchecked::<ArchivedAdj>(&self.p2f_adj_mmap[..]) };
        if (parent_node_idx as usize) >= p2f.adj.len() {
            return vec![];
        }
        p2f.adj[parent_node_idx as usize]
            .iter()
            .map(|e| { let i: i32 = e.node_idx.into(); i as u32 })
            .collect()
    }

    fn child_to_parents(&self, child_node_idx: u32) -> Vec<u32> {
        if (child_node_idx as usize) >= self.f2p_adj.len() {
            return vec![];
        }
        self.f2p_adj[child_node_idx as usize]
            .iter()
            .map(|&x| x as u32)
            .collect()
    }

    // Returns flat array shape (N*3,) — reshape to (-1, 3) in Python.
    // Chain A→B→C: A.f2p_edges contains B, B.f2p_edges contains C.
    fn enumerate_chain_triads<'py>(
        &self,
        py: Python<'py>,
        a_table: &str,
        b_table: &str,
        c_table: &str,
        max_samples: usize,
    ) -> PyResult<PyObject> {
        let (a_off, a_cnt) = self.table_range(a_table);
        let (b_off, b_cnt) = self.table_range(b_table);
        let (c_off, c_cnt) = self.table_range(c_table);

        let mut results: Vec<u32> = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        let a_total = a_cnt as usize;
        let a_sample: Vec<i32> = if a_total > max_samples * 20 {
            index::sample(&mut rng, a_total, max_samples * 20)
                .into_vec()
                .into_iter()
                .map(|i| a_off + i as i32)
                .collect()
        } else {
            (a_off..a_off + a_cnt).collect()
        };

        'outer: for a_idx in a_sample {
            let node_a = self.get_node(a_idx);
            for edge_ab in node_a.f2p_edges.iter() {
                let b_idx: i32 = edge_ab.node_idx.into();
                if b_idx < b_off || b_idx >= b_off + b_cnt {
                    continue;
                }
                let node_b = self.get_node(b_idx);
                for edge_bc in node_b.f2p_edges.iter() {
                    let c_idx: i32 = edge_bc.node_idx.into();
                    if c_idx >= c_off && c_idx < c_off + c_cnt {
                        results.extend_from_slice(&[a_idx as u32, b_idx as u32, c_idx as u32]);
                        if results.len() / 3 >= max_samples {
                            break 'outer;
                        }
                    }
                }
            }
        }

        PyArray1::from_vec(py, results).into_py_any(py)
    }

    // Returns flat array shape (N*3,) — reshape to (-1, 3) in Python.
    // V-shape A→B←C: both A and C have f2p_edges to B.
    fn enumerate_v_triads<'py>(
        &self,
        py: Python<'py>,
        a_table: &str,
        b_table: &str,
        c_table: &str,
        max_samples: usize,
    ) -> PyResult<PyObject> {
        let (a_off, a_cnt) = self.table_range(a_table);
        let (b_off, b_cnt) = self.table_range(b_table);
        let (c_off, c_cnt) = self.table_range(c_table);

        let p2f = unsafe { rkyv::access_unchecked::<ArchivedAdj>(&self.p2f_adj_mmap[..]) };
        let mut results: Vec<u32> = Vec::new();

        'outer: for b_idx in b_off..b_off + b_cnt {
            if (b_idx as usize) >= p2f.adj.len() {
                continue;
            }
            let children = &p2f.adj[b_idx as usize];

            let a_children: Vec<i32> = children
                .iter()
                .filter_map(|e| {
                    let idx: i32 = e.node_idx.into();
                    if idx >= a_off && idx < a_off + a_cnt { Some(idx) } else { None }
                })
                .collect();

            let c_children: Vec<i32> = children
                .iter()
                .filter_map(|e| {
                    let idx: i32 = e.node_idx.into();
                    if idx >= c_off && idx < c_off + c_cnt { Some(idx) } else { None }
                })
                .collect();

            for &a_idx in &a_children {
                for &c_idx in &c_children {
                    if a_idx == c_idx {
                        continue;
                    }
                    results.extend_from_slice(&[a_idx as u32, b_idx as u32, c_idx as u32]);
                    if results.len() / 3 >= max_samples {
                        break 'outer;
                    }
                }
            }
        }

        PyArray1::from_vec(py, results).into_py_any(py)
    }
}

impl GraphInspector {
    fn get_node(&self, idx: i32) -> &ArchivedNode {
        let l = self.offsets[idx as usize] as usize;
        let r = self.offsets[(idx + 1) as usize] as usize;
        unsafe { rkyv::access_unchecked::<ArchivedNode>(&self.mmap[l..r]) }
    }

    fn table_for_node(&self, node_idx: i32) -> Option<String> {
        self.tables
            .iter()
            .find(|t| node_idx >= t.offset && node_idx < t.offset + t.count)
            .map(|t| t.name.clone())
    }

    fn table_range(&self, table_name: &str) -> (i32, i32) {
        self.tables
            .iter()
            .find(|t| t.name == table_name)
            .map_or((0, 0), |t| (t.offset, t.count))
    }
}
