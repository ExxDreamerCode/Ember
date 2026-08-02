use super::Searcher;
use crate::board::{BoardState, MAX_PLY};
use crate::nnue::{NNUEAccumulator, NNUENet, NNUEThreatAccumulator, NnueBackend};

#[derive(Clone, Copy)]
pub(super) struct ClassicEval;

#[derive(Clone, Copy)]
pub(super) struct NnueEval<'a, B: NnueBackend> {
    pub(super) net: &'a NNUENet,
    pub(super) _backend: B,
}

#[derive(Clone, Copy)]
pub(super) struct ThreatNnueEval<'a, B: NnueBackend> {
    pub(super) net: &'a NNUENet,
    pub(super) _backend: B,
}

pub(super) trait SearchEval: Copy {
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32;

    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn push_acc(
        self,
        searcher: &mut Searcher,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    );

    fn ensure_child_stack(self, searcher: &mut Searcher, ply: usize);

    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize);
}

impl SearchEval for ClassicEval {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        _ply: usize,
    ) -> i32 {
        searcher.static_eval_classic::<CHESS960>(st)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_classic::<CHESS960>(st)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_acc(
        self,
        _searcher: &mut Searcher,
        _st_before: &BoardState,
        _st_after: &BoardState,
        _sr: usize,
        _sc: usize,
        _er: usize,
        _ec: usize,
        _promotion: u8,
        _ply: usize,
    ) {
    }

    #[inline(always)]
    fn ensure_child_stack(self, _searcher: &mut Searcher, _ply: usize) {}

    #[inline(always)]
    fn copy_null_acc(self, _searcher: &mut Searcher, _ply: usize) {}
}

impl<'a, B: NnueBackend> SearchEval for NnueEval<'a, B> {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32 {
        searcher.static_eval_nnue::<CHESS960, B>(st, ply, self.net)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_nnue::<CHESS960, B>(st, self.net)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_acc(
        self,
        searcher: &mut Searcher,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        searcher.push_nnue_acc::<B>(
            self.net, st_before, st_after, sr, sc, er, ec, promotion, ply,
        );
    }

    #[inline(always)]
    fn ensure_child_stack(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 >= searcher.nnue_stack.len() && ply + 1 < MAX_PLY + 1 {
            searcher
                .nnue_stack
                .resize(ply + 2, NNUEAccumulator::new(self.net.hidden_size));
        }
    }

    #[inline(always)]
    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 < searcher.nnue_stack.len() {
            let (left, right) = searcher.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
        }
    }
}

impl<'a, B: NnueBackend> SearchEval for ThreatNnueEval<'a, B> {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32 {
        searcher.static_eval_threat_nnue::<CHESS960, B>(st, ply, self.net)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_threat_nnue::<CHESS960, B>(st, self.net)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_acc(
        self,
        searcher: &mut Searcher,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        searcher.push_threat_nnue_acc::<B>(
            self.net, st_before, st_after, sr, sc, er, ec, promotion, ply,
        );
    }

    #[inline(always)]
    fn ensure_child_stack(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 >= searcher.nnue_stack.len() && ply + 1 < MAX_PLY + 1 {
            searcher
                .nnue_stack
                .resize(ply + 2, NNUEAccumulator::new(self.net.hidden_size));
        }
        if ply + 1 >= searcher.threat_stack.len() && ply + 1 < MAX_PLY + 1 {
            searcher
                .threat_stack
                .resize(ply + 2, NNUEThreatAccumulator::new(self.net.hidden_size));
        }
    }

    #[inline(always)]
    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 < searcher.nnue_stack.len() {
            let (left, right) = searcher.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
        }
        if ply + 1 < searcher.threat_stack.len() {
            let (left, right) = searcher.threat_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
        }
    }
}
