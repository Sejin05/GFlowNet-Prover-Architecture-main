#![deny(clippy::unwrap_used, clippy::expect_used, unsafe_op_in_unsafe_fn)]

//! 6-layer State-Graph Relational Graph Attention Network (`StateGraphRgatEncoder`).
//!
//! Encodes the heterogeneous state-dependency graph (HSDG) updated at every transaction step $t$
//! by `RevmArena`. Projects node and edge features, runs multi-head relational attention across
//! all 8 canonical edge relations, and performs global context pooling $g_t \in \mathbb{R}^{B \times 512}$.

use gfn_common::error::{GfnError, GfnResult};
use gfn_extractor::HeteroGraphTensors;
use tch::nn::{self, Module};
use tch::{Kind, Tensor};

/// Single Relational Graph Attention Network (RGAT) Convolutional Layer.
///
/// Implements relational multi-head sparse scatter-attention across 8 edge relation types
/// connecting Storage, Function, Condition, and ExternalCall nodes.
pub struct RgatConvLayer {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    edge_proj: Vec<nn::Linear>,
    out_proj: nn::Linear,
    storage_ln: nn::LayerNorm,
    func_ln: nn::LayerNorm,
    cond_ln: nn::LayerNorm,
    ext_ln: nn::LayerNorm,
    hidden_dim: i64,
    num_heads: i64,
}

impl RgatConvLayer {
    /// Creates a new `RgatConvLayer` module within the given VarStore path.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_heads: i64) -> Self {
        let q_proj = nn::linear(vs / "q_proj", hidden_dim, hidden_dim, Default::default());
        let k_proj = nn::linear(vs / "k_proj", hidden_dim, hidden_dim, Default::default());
        let v_proj = nn::linear(vs / "v_proj", hidden_dim, hidden_dim, Default::default());

        let mut edge_proj = Vec::with_capacity(8);
        for r in 0..8 {
            edge_proj.push(nn::linear(
                vs / format!("edge_proj_{r}"),
                16,
                hidden_dim,
                Default::default(),
            ));
        }

        let out_proj = nn::linear(vs / "out_proj", hidden_dim, hidden_dim, Default::default());
        let storage_ln = nn::layer_norm(vs / "storage_ln", vec![hidden_dim], Default::default());
        let func_ln = nn::layer_norm(vs / "func_ln", vec![hidden_dim], Default::default());
        let cond_ln = nn::layer_norm(vs / "cond_ln", vec![hidden_dim], Default::default());
        let ext_ln = nn::layer_norm(vs / "ext_ln", vec![hidden_dim], Default::default());

        Self {
            q_proj,
            k_proj,
            v_proj,
            edge_proj,
            out_proj,
            storage_ln,
            func_ln,
            cond_ln,
            ext_ln,
            hidden_dim,
            num_heads,
        }
    }

    /// Computes relational message passing across the 8 edge relations.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        storage_h: &Tensor,
        function_h: &Tensor,
        condition_h: &Tensor,
        ext_call_h: &Tensor,
        edge_indices: &[Tensor; 8],
        edge_attrs: &[Tensor; 8],
        train: bool,
    ) -> GfnResult<(Tensor, Tensor, Tensor, Tensor)> {
        let head_dim = (self.hidden_dim / self.num_heads).max(1) as f64;
        let attn_scale = head_dim.sqrt();

        // Project Q, K, V for each node type
        let s_q = self.q_proj.forward(storage_h);
        let s_k = self.k_proj.forward(storage_h);
        let s_v = self.v_proj.forward(storage_h);

        let f_q = self.q_proj.forward(function_h);
        let f_k = self.k_proj.forward(function_h);
        let f_v = self.v_proj.forward(function_h);

        let c_q = self.q_proj.forward(condition_h);
        let c_k = self.k_proj.forward(condition_h);
        let c_v = self.v_proj.forward(condition_h);

        let e_q = self.q_proj.forward(ext_call_h);
        let e_k = self.k_proj.forward(ext_call_h);
        let e_v = self.v_proj.forward(ext_call_h);

        // Accumulators for incoming messages
        let mut s_msg = Tensor::zeros(
            storage_h.size().as_slice(),
            (Kind::Float, storage_h.device()),
        );
        let mut f_msg = Tensor::zeros(
            function_h.size().as_slice(),
            (Kind::Float, function_h.device()),
        );
        let mut c_msg = Tensor::zeros(
            condition_h.size().as_slice(),
            (Kind::Float, condition_h.device()),
        );
        let mut e_msg = Tensor::zeros(
            ext_call_h.size().as_slice(),
            (Kind::Float, ext_call_h.device()),
        );

        // Process each of the 8 canonical HSDG relations
        for r in 0..8 {
            let edge_index = &edge_indices[r];
            let edge_attr = &edge_attrs[r];

            if edge_index.size().len() < 2 || edge_index.size()[1] == 0 {
                continue;
            }

            let num_edges = edge_index.size()[1];
            if num_edges == 0 {
                continue;
            }

            let src_idx = edge_index.select(0, 0);
            let dst_idx = edge_index.select(0, 1);

            // Determine source and destination node representations according to relation r
            let (src_k, src_v, dst_q, num_dst_nodes, dst_target) = match r {
                0 => (&f_k, &f_v, &s_q, storage_h.size()[0], 0), // ReadsFrom: Func -> Storage
                1 => (&f_k, &f_v, &s_q, storage_h.size()[0], 0), // WritesTo: Func -> Storage
                2 => (&c_k, &c_v, &f_q, function_h.size()[0], 1), // Guards: Cond -> Func
                3 => (&f_k, &f_v, &e_q, ext_call_h.size()[0], 3), // Invokes: Func -> ExtCall
                4 => (&s_k, &s_v, &f_q, function_h.size()[0], 1), // RevReadsFrom: Storage -> Func
                5 => (&s_k, &s_v, &f_q, function_h.size()[0], 1), // RevWritesTo: Storage -> Func
                6 => (&f_k, &f_v, &c_q, condition_h.size()[0], 2), // RevGuards: Func -> Cond
                7 => (&e_k, &e_v, &f_q, function_h.size()[0], 1), // RevInvokes: ExtCall -> Func
                _ => (&f_k, &f_v, &s_q, storage_h.size()[0], 0),
            };

            if src_k.size()[0] == 0 || dst_q.size()[0] == 0 || num_dst_nodes == 0 {
                continue;
            }

            let edge_e = self.edge_proj[r].forward(edge_attr);

            // Gather source K, V and destination Q for each edge
            let edge_src_k = src_k.index_select(0, &src_idx);
            let edge_src_v = src_v.index_select(0, &src_idx);
            let edge_dst_q = dst_q.index_select(0, &dst_idx);

            // Compute attention score per edge: (dst_q * (src_k + edge_e)).sum(-1) / sqrt(d_k), clamped for stability
            let raw_scores = ((&edge_dst_q * (&edge_src_k + &edge_e)).sum_dim_intlist(
                Some([-1].as_slice()),
                false,
                Kind::Float,
            ) / attn_scale)
                .clamp(-30.0, 30.0);

            // Numerically stable segment-softmax
            let max_score = raw_scores.max();
            let exp_scores = (raw_scores - max_score).exp();

            let mut denom = Tensor::zeros([num_dst_nodes], (Kind::Float, storage_h.device()));
            let _ = denom.scatter_add_(0, &dst_idx, &exp_scores);
            let gathered_denom = denom.index_select(0, &dst_idx) + 1e-6;
            let alpha = &exp_scores / &gathered_denom;

            let weighted_v = &edge_src_v * &alpha.unsqueeze(-1);

            let mut aggregated_msg = Tensor::zeros(
                [num_dst_nodes, self.hidden_dim],
                (Kind::Float, storage_h.device()),
            );
            let expanded_dst_idx = dst_idx.unsqueeze(-1).expand([-1, self.hidden_dim], false);
            let _ = aggregated_msg.scatter_add_(0, &expanded_dst_idx, &weighted_v);

            match dst_target {
                0 => s_msg += aggregated_msg,
                1 => f_msg += aggregated_msg,
                2 => c_msg += aggregated_msg,
                3 => e_msg += aggregated_msg,
                _ => {}
            }
        }

        // Apply GELU, Dropout, Residual connection, and LayerNorm for each node type
        let apply_update = |h: &Tensor, msg: &Tensor, ln: &nn::LayerNorm| -> Tensor {
            let mut act = msg.gelu("none");
            if train {
                act = act.dropout(0.1, true);
            }
            let out = self.out_proj.forward(&act);
            ln.forward(&(h + out))
        };

        let new_storage = apply_update(storage_h, &s_msg, &self.storage_ln);
        let new_func = apply_update(function_h, &f_msg, &self.func_ln);
        let new_cond = apply_update(condition_h, &c_msg, &self.cond_ln);
        let new_ext = apply_update(ext_call_h, &e_msg, &self.ext_ln);

        Ok((new_storage, new_func, new_cond, new_ext))
    }
}

/// 6-Layer Relational Graph Attention Network Encoder (`StateGraphRgatEncoder`).
pub struct StateGraphRgatEncoder {
    storage_in_proj: nn::Linear,
    func_in_proj: nn::Linear,
    cond_in_proj: nn::Linear,
    ext_in_proj: nn::Linear,
    env_in_proj: nn::Linear,
    layers: Vec<RgatConvLayer>,
    ctx_mlp: nn::Sequential,
    hidden_dim: i64,
    num_layers: usize,
    num_heads: i64,
}

impl StateGraphRgatEncoder {
    /// Constructs a new `StateGraphRgatEncoder` with type-specific projections and 6 RGAT layers.
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_layers: usize, num_heads: i64) -> Self {
        let storage_in_proj = nn::linear(vs / "storage_in", 64, hidden_dim, Default::default());
        let func_in_proj = nn::linear(vs / "func_in", 64, hidden_dim, Default::default());
        let cond_in_proj = nn::linear(vs / "cond_in", 48, hidden_dim, Default::default());
        let ext_in_proj = nn::linear(vs / "ext_in", 32, hidden_dim, Default::default());
        let env_in_proj = nn::linear(vs / "env_in", 32, hidden_dim, Default::default());

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            layers.push(RgatConvLayer::new(
                &(vs / format!("rgat_layer_{l}")),
                hidden_dim,
                num_heads,
            ));
        }

        let ctx_mlp = nn::seq()
            .add(nn::linear(
                vs / "ctx_mlp_1",
                hidden_dim * 4,
                hidden_dim * 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::layer_norm(
                vs / "ctx_ln_1",
                vec![hidden_dim * 2],
                Default::default(),
            ))
            .add(nn::linear(
                vs / "ctx_mlp_2",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.gelu("none"))
            .add(nn::layer_norm(
                vs / "ctx_ln_2",
                vec![hidden_dim],
                Default::default(),
            ));

        Self {
            storage_in_proj,
            func_in_proj,
            cond_in_proj,
            ext_in_proj,
            env_in_proj,
            layers,
            ctx_mlp,
            hidden_dim,
            num_layers,
            num_heads,
        }
    }

    /// Returns the hidden dimension of the encoder.
    #[must_use]
    pub fn hidden_dim(&self) -> i64 {
        self.hidden_dim
    }

    /// Returns the number of layers in the encoder.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns the number of attention heads in the encoder.
    #[must_use]
    pub fn num_heads(&self) -> i64 {
        self.num_heads
    }

    /// Forward pass through the 6-layer RGAT encoder.
    ///
    /// Returns:
    /// - `global_ctx`: Shape `[B, hidden_dim]` in `Kind::Float`
    /// - `func_embeddings`: Shape `[B, N_func, hidden_dim]` in `Kind::Float`
    /// - `storage_embeddings`: Shape `[B, N_storage, hidden_dim]` in `Kind::Float`
    pub fn forward_t(
        &self,
        batched_graph: &HeteroGraphTensors,
        env_context: &Tensor,
        train: bool,
    ) -> GfnResult<(Tensor, Tensor, Tensor)> {
        let batch_size = env_context.size()[0];
        if batch_size <= 0 {
            return Err(GfnError::Tensor {
                message: "Batch size in env_context must be greater than 0".to_string(),
            });
        }

        // Project raw node features to hidden_dim
        let mut storage_h = self
            .storage_in_proj
            .forward(&batched_graph.storage_x)
            .gelu("none");
        let mut func_h = self
            .func_in_proj
            .forward(&batched_graph.function_x)
            .gelu("none");
        let mut cond_h = self
            .cond_in_proj
            .forward(&batched_graph.condition_x)
            .gelu("none");
        let mut ext_h = self
            .ext_in_proj
            .forward(&batched_graph.ext_call_x)
            .gelu("none");

        // Pass through stacked RGAT convolutional layers
        for layer in &self.layers {
            let (next_s, next_f, next_c, next_e) = layer.forward(
                &storage_h,
                &func_h,
                &cond_h,
                &ext_h,
                &batched_graph.edge_indices,
                &batched_graph.edge_attrs,
                train,
            )?;
            storage_h = next_s;
            func_h = next_f;
            cond_h = next_c;
            ext_h = next_e;
        }

        // Global context pooling per graph in batch:
        // SegmentMean(H_storage) || SegmentMax(H_cond) || SegmentMean(H_func) || W_env c_t
        let n_storage_total = storage_h.size()[0];
        let n_func_total = func_h.size()[0];
        let n_cond_total = cond_h.size()[0];

        let n_storage_per_b = n_storage_total / batch_size;
        let n_func_per_b = n_func_total / batch_size;

        let storage_batched = if n_storage_per_b > 0 {
            storage_h.view([batch_size, n_storage_per_b, self.hidden_dim])
        } else {
            Tensor::zeros(
                [batch_size, 1, self.hidden_dim],
                (Kind::Float, storage_h.device()),
            )
        };

        let func_batched = if n_func_per_b > 0 {
            func_h.view([batch_size, n_func_per_b, self.hidden_dim])
        } else {
            Tensor::zeros(
                [batch_size, 1, self.hidden_dim],
                (Kind::Float, func_h.device()),
            )
        };

        let pooled_storage = storage_batched.mean_dim(Some([1].as_slice()), false, Kind::Float);
        let pooled_func = func_batched.mean_dim(Some([1].as_slice()), false, Kind::Float);

        let pooled_cond = if n_cond_total > 0 && (n_cond_total / batch_size) > 0 {
            let n_cond_per_b = n_cond_total / batch_size;
            let cond_batched = cond_h.view([batch_size, n_cond_per_b, self.hidden_dim]);
            cond_batched.max_dim(1, false).0
        } else {
            Tensor::zeros(
                [batch_size, self.hidden_dim],
                (Kind::Float, storage_h.device()),
            )
        };

        let env_proj = self.env_in_proj.forward(env_context).gelu("none");

        // Concatenate pooled representations into [B, hidden_dim * 4]
        let pooled_concat =
            Tensor::cat(&[&pooled_storage, &pooled_cond, &pooled_func, &env_proj], 1);

        let global_ctx = self.ctx_mlp.forward(&pooled_concat).to_kind(Kind::Float);
        let out_func_embeddings = func_batched.to_kind(Kind::Float);
        let out_storage_embeddings = storage_batched.to_kind(Kind::Float);

        Ok((global_ctx, out_func_embeddings, out_storage_embeddings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_rgat_forward_shapes_and_finiteness() {
        let vs = nn::VarStore::new(Device::Cpu);
        let encoder = StateGraphRgatEncoder::new(&vs.root(), 64, 2, 4);

        let storage_x = Tensor::zeros([4, 64], (Kind::Float, Device::Cpu));
        let function_x = Tensor::zeros([6, 64], (Kind::Float, Device::Cpu));
        let condition_x = Tensor::zeros([2, 48], (Kind::Float, Device::Cpu));
        let ext_call_x = Tensor::zeros([2, 32], (Kind::Float, Device::Cpu));

        let edge_indices = [
            Tensor::from_slice2(&[[0i64, 1], [0i64, 1]]),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 0], (Kind::Int64, Device::Cpu)),
        ];

        let edge_attrs = [
            Tensor::zeros([2, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
            Tensor::zeros([0, 16], (Kind::Float, Device::Cpu)),
        ];

        let graph = HeteroGraphTensors {
            storage_x,
            function_x,
            condition_x,
            ext_call_x,
            edge_indices,
            edge_attrs,
        };

        let env_context = Tensor::zeros([2, 32], (Kind::Float, Device::Cpu));

        let res = encoder.forward_t(&graph, &env_context, false);
        assert!(res.is_ok());
        if let Ok((global_ctx, func_emb, storage_emb)) = res {
            assert_eq!(global_ctx.size(), vec![2, 64]);
            assert_eq!(func_emb.size(), vec![2, 3, 64]);
            assert_eq!(storage_emb.size(), vec![2, 2, 64]);
            assert_eq!(global_ctx.kind(), Kind::Float);
            assert_eq!(global_ctx.isfinite().all().int64_value(&[]), 1);
            assert_eq!(func_emb.isfinite().all().int64_value(&[]), 1);
            assert_eq!(storage_emb.isfinite().all().int64_value(&[]), 1);
        }
    }
}
