use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets,
};
use half::bf16;
use itertools::izip;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use std::collections::HashMap;
use std::env::var;
use std::fs;
use std::io::{BufReader, Read};
use std::str;

const MAX_F2P_NBRS: usize = 5;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    f2p_rel_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_padding: Vec<bool>,
    true_batch_size: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    f2p_rel_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_padding: &'a mut [bool],
}

impl Vecs {
    fn new(batch_size: usize, seq_len: usize, true_batch_size: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            f2p_rel_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            datetime_values: vec![bf16::ZERO; l],
            boolean_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_padding: vec![true; l],
            true_batch_size,
        }
    }

    fn chunks_exact_mut(&mut self, seq_len: usize, d_text: usize) -> impl Iterator<Item = Slices> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.f2p_rel_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                f2p_rel_idxs,
                table_name_idxs,
                col_name_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_padding,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                f2p_rel_idxs,
                table_name_idxs,
                col_name_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_padding,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_rel_idxs", PyArray1::from_vec(py, self.f2p_rel_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("masks", PyArray1::from_vec(py, self.masks))
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("true_batch_size", self.true_batch_size)
                .into_py_any(py)
                .unwrap(),
        ])
    }
}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    relation_text_idx_to_idx: HashMap<i32, i32>,
    offsets: Vec<i64>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    seq_len: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    max_bfs_width: usize,
    epoch: u64,
    d_text: usize,
    seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        seq_len: usize,
        max_bfs_width: usize,
        embedding_model: &str,
        d_text: usize,
        seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        relation_table_offsets: Vec<i32>,
    ) -> Self {
        let mut datasets = Vec::new();
        let mut items = Vec::new();
        assert_eq!(relation_table_offsets.len(), dataset_tuples.len());
        for (i, (db_name, node_idx_offset, num_nodes)) in dataset_tuples.into_iter().enumerate() {
            let relation_table_offset = relation_table_offsets[i];
            let pre_root = var("NANORT_PRE_ROOT")
                .unwrap_or_else(|_| format!("{}/.cache/modded-nanort/pre", var("HOME").unwrap()));
            let pre_path = format!("{}/{}", pre_root, db_name);
            let nodes_path = format!("{}/nodes.rkyv", pre_path);
            let file = fs::File::open(&nodes_path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };

            let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model);
            let text_file = fs::File::open(&text_path).unwrap();
            let text_mmap = unsafe { Mmap::map(&text_file).unwrap() };

            let offsets_path = format!("{}/offsets.rkyv", pre_path);
            let file = fs::File::open(&offsets_path).unwrap();
            let mut bytes = Vec::new();
            BufReader::new(file).read_to_end(&mut bytes).unwrap();
            let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
            let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
            let offsets = offsets.offsets;

            let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
            let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
            let p2f_adj_mmap = unsafe { Mmap::map(&p2f_adj_file).unwrap() };

            let relation_text_idxs_path = format!("{}/relation_text_idxs.json", pre_path);
            let relation_text_idxs_file = fs::File::open(&relation_text_idxs_path).unwrap();
            let relation_text_idxs: Vec<i32> =
                serde_json::from_reader(relation_text_idxs_file).unwrap();
            let relation_text_idx_to_idx = relation_text_idxs
                .into_iter()
                .enumerate()
                .map(|(j, text_idx)| (text_idx, relation_table_offset + j as i32))
                .collect();
            let target = target_columns[i];

            datasets.push(Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                relation_text_idx_to_idx,
                offsets,
            });

            for j in node_idx_offset..node_idx_offset + num_nodes {
                let node = get_node(&datasets[i], j);
                // skip the node if target column was removed during preprocessing
                if node.col_name_idxs.iter().any(|&c| c == target) {
                    items.push(Item {
                        dataset_idx: i as i32,
                        node_idx: j,
                    });
                }
            }
        }

        let epoch = 0;
        Self {
            batch_size,
            seq_len,
            datasets,
            items,
            max_bfs_width,
            epoch,
            d_text,
            seed,
            target_columns,
            columns_to_drop,
        }
    }

    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn batch_py<'a>(&self, py: Python<'a>, batch_idx: usize) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

impl Sampler {
    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size)
    }

    fn batch(&self, batch_idx: usize) -> Vecs {
        let true_batch_size = self
            .batch_size
            .min(self.items.len() - batch_idx * self.batch_size);

        let mut vecs = Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);
        vecs.chunks_exact_mut(self.seq_len, self.d_text)
            .enumerate()
            .for_each(|(i, slices)| {
                let j = batch_idx * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                self.seq(item, slices);
            });
        vecs
    }

    fn seq(&self, item: &Item, slices: Slices) {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let target_column = self.target_columns[item.dataset_idx as usize];
        //define let columns to drop which is a vector of i32 and is at the same index as target_columns
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];
        let seed_node_idx = item.node_idx;

        let mut visited = vec![false; dataset.offsets.len() - 1];

        let mut f2p_ftr = vec![(0, seed_node_idx)];
        let seed_node = get_node(dataset, seed_node_idx);
        let mut p2f_ftr = Vec::<Vec<_>>::new();

        let mut seq_i = 0;
        let mut rng = StdRng::seed_from_u64(
            self.epoch
                .wrapping_add(seed_node_idx as u64)
                .wrapping_add(self.seed),
        );
        loop {
            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for (i, node) in p2f_ftr.iter().enumerate() {
                    if !node.is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    return;
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    p2f_ftr[depth].swap(r, l - 1);
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            visited[node_idx as usize] = true;

            let node = get_node(dataset, node_idx);

            for edge in node.f2p_edges.iter() {
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // temporary storage for db edges to be subsampled
            let mut db_p2f_ftr: Vec<i32> = Vec::new();

            for edge in p2f_edges.iter() {
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != seed_node.table_name_idx
                    && edge.table_type != ArchivedTableType::Db
                {
                    continue;
                }

                // temporal constraint
                if edge.timestamp.is_some()
                    && seed_node.timestamp.is_some()
                    && edge.timestamp > seed_node.timestamp
                {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push(edge.node_idx.into());
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            let idxs = if db_p2f_ftr.len() > self.max_bfs_width {
                index::sample(&mut rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(db_p2f_ftr[*idx]);
            }

            let num_cells = node.col_name_idxs.len();
            for cell_i in 0..num_cells {
                let col_idx: i32 = node.col_name_idxs[cell_i].into();
                if (node.node_idx == seed_node_idx && columns_to_drop.contains(&col_idx))
                    || (node.timestamp == seed_node.timestamp && columns_to_drop.contains(&col_idx))
                {
                    continue; // do not add this cell to the sequence
                }
                slices.node_idxs[seq_i] = node.node_idx.into();

                assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
                assert_eq!(node.f2p_nbr_idxs.len(), node.f2p_edges.len());
                for (j, (f2p_nbr_idx, f2p_edge)) in node
                    .f2p_nbr_idxs
                    .iter()
                    .zip(node.f2p_edges.iter())
                    .enumerate()
                {
                    slices.f2p_nbr_idxs[seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
                    let relation_text_idx: i32 = f2p_edge.relation_text_idx.into();
                    slices.f2p_rel_idxs[seq_i * MAX_F2P_NBRS + j] = *dataset
                        .relation_text_idx_to_idx
                        .get(&relation_text_idx)
                        .unwrap();
                }

                slices.table_name_idxs[seq_i] = node.table_name_idx.into();
                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();
                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(
                        dataset,
                        slices.col_name_idxs[seq_i],
                        self.d_text,
                    ));

                let s = node.sem_types[cell_i].clone() as i32;
                slices.sem_types[seq_i] = s;

                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());

                let text_idx: i32 = node.text_values[cell_i].into();
                slices.text_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

                slices.datetime_values[seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

                slices.boolean_values[seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

                let is_target =
                    seed_node_idx == node.node_idx && node.col_name_idxs[cell_i] == target_column;
                slices.is_targets[seq_i] = is_target;

                slices.masks[seq_i] = is_target;

                slices.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }
            if seq_i >= self.seq_len {
                break;
            }
        }
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Adj, Edge, Node, SemType, TableType};

    #[test]
    fn frozen_graph_sampling_preserves_targets_relations_and_partial_batches() {
        let directory = std::env::temp_dir().join(format!(
            "nanort-sampler-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let edge = |node_idx| Edge {
            node_idx,
            table_name_idx: 1,
            relation_text_idx: 2,
            table_type: TableType::Db,
            timestamp: None,
        };
        let mut nodes = Vec::new();
        let mut offsets = vec![0];
        for node_idx in 0..2 {
            let node = Node {
                node_idx,
                f2p_nbr_idxs: if node_idx == 0 { vec![1] } else { vec![] },
                f2p_edges: if node_idx == 0 { vec![edge(1)] } else { vec![] },
                table_name_idx: 1,
                col_name_idxs: vec![0],
                sem_types: vec![SemType::Boolean],
                number_values: vec![0.0],
                text_values: vec![1],
                datetime_values: vec![0.0],
                boolean_values: vec![if node_idx == 0 { 1.0 } else { 0.0 }],
                class_value_idx: vec![-1],
                ..Node::default()
            };
            nodes.extend_from_slice(&rkyv::to_bytes::<Error>(&node).unwrap());
            offsets.push(nodes.len() as i64);
        }
        fs::write(directory.join("nodes"), nodes).unwrap();
        let adjacency = Adj {
            adj: vec![vec![], vec![edge(0)]],
        };
        fs::write(
            directory.join("adj"),
            rkyv::to_bytes::<Error>(&adjacency).unwrap(),
        )
        .unwrap();
        let text: Vec<u8> = (0..12)
            .flat_map(|value| bf16::from_f32(value as f32).to_le_bytes())
            .collect();
        fs::write(directory.join("text"), text).unwrap();
        let mmap =
            |name| unsafe { Mmap::map(&fs::File::open(directory.join(name)).unwrap()).unwrap() };
        let make_sampler = || Sampler {
            batch_size: 3,
            seq_len: 4,
            datasets: vec![Dataset {
                mmap: mmap("nodes"),
                text_mmap: mmap("text"),
                p2f_adj_mmap: mmap("adj"),
                relation_text_idx_to_idx: HashMap::from([(2, 0)]),
                offsets: offsets.clone(),
            }],
            items: (0..2)
                .map(|node_idx| Item {
                    dataset_idx: 0,
                    node_idx,
                })
                .collect(),
            max_bfs_width: 256,
            epoch: 0,
            d_text: 4,
            seed: 3,
            target_columns: vec![0],
            columns_to_drop: vec![vec![]],
        };
        let mut sampler = make_sampler();
        assert_eq!(sampler.len(), 1);
        let batch = sampler.batch(0);
        assert_eq!(batch.true_batch_size, 2);
        assert_eq!(&batch.node_idxs[..4], &[0, 1, -1, -1]);
        assert_eq!(&batch.masks[..4], &[true, false, false, false]);
        assert_eq!(batch.masks, batch.is_targets);
        assert_eq!(&batch.is_padding[..4], &[false, false, true, true]);
        assert_eq!(batch.f2p_nbr_idxs[0], 1);
        assert_eq!(batch.f2p_rel_idxs[0], 0);
        assert_eq!(batch.boolean_values[0], bf16::ONE);
        assert_eq!(
            &batch.col_name_values[..4],
            &[0., 1., 2., 3.].map(bf16::from_f32)
        );
        // The unused final batch slot repeats an example, as before; the
        // evaluator excludes it using true_batch_size.
        assert_eq!(&batch.node_idxs[8..12], &batch.node_idxs[..4]);
        let mut same_seed = make_sampler();
        sampler.shuffle_py(2);
        same_seed.shuffle_py(2);
        assert_eq!(sampler.batch(0).node_idxs, same_seed.batch(0).node_idxs);
        fs::remove_dir_all(directory).unwrap();
    }
}
