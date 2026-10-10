// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime dose for the living-weights steering site
//! (`crate::living::steer`), on every rank of a pair.
//!
//! Rank 0 (the scheduler, at a quiescent point) sends `EP_CMD_LIVING_STEER`
//! with the dose's F32 bits and the flush flag; the worker applies the same.
//! Each rank writes its own device control word between steps, so the CUDA
//! graphs both captured at startup keep replaying unchanged.

use anyhow::{Result, ensure};

use super::types::TransformerModel;
use crate::traits::Model;

/// Whole-worker command: dose bits, flush flag.
pub(super) const EP_CMD_LIVING_STEER: u32 = 0xFFFF_FFED;

impl TransformerModel {
    /// Head side: validate, tell the worker, then apply here.
    pub(super) fn living_steer_set(&self, dose: f32, flush_prefix_cache: bool) -> Result<()> {
        ensure!(
            crate::living::steer::loaded(),
            "no steering vectors are loaded (start with ATLAS_STEER_DIR)"
        );
        ensure!(dose.is_finite(), "steering dose {dose} is not finite");
        if self.multi_rank_protocol_active() {
            self.ep_broadcast_seq_and_cmd(0, EP_CMD_LIVING_STEER, self.ep_protocol_v2)?;
            self.ep_broadcast_u32(dose.to_bits())?;
            self.ep_broadcast_u32(u32::from(flush_prefix_cache))?;
        }
        self.living_steer_apply(dose, flush_prefix_cache)
    }

    /// Worker side of [`EP_CMD_LIVING_STEER`].
    pub(super) fn living_steer_receive(&self) -> Result<bool> {
        let dose = f32::from_bits(self.ep_broadcast_u32(0)?);
        let flush = self.ep_broadcast_u32(0)? != 0;
        self.living_steer_apply(dose, flush)?;
        Ok(true)
    }

    fn living_steer_apply(&self, dose: f32, flush_prefix_cache: bool) -> Result<()> {
        self.gpu.synchronize_device()?;
        crate::living::steer::set_dose(self.gpu.as_ref(), dose)?;
        if flush_prefix_cache {
            let freed = Model::reclaim_prefix_blocks(self, usize::MAX);
            tracing::info!("steer: prefix cache flushed ({freed} blocks freed)");
        }
        Ok(())
    }
}
