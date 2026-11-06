//! Stacker stub for pure Rust environments without raw-dylib toolchain requirements.

#[inline(always)]
pub fn maybe_grow<R, F: FnOnce() -> R>(_red_zone: usize, _stack_size: usize, callback: F) -> R {
    callback()
}

#[inline(always)]
pub fn grow<R, F: FnOnce() -> R>(_stack_size: usize, callback: F) -> R {
    callback()
}

#[inline(always)]
pub fn remaining_stack() -> Option<usize> {
    None
}
