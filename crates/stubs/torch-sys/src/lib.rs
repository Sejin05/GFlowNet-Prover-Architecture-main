#![allow(warnings)]
pub use c_generated::*;
pub mod cuda;
pub mod io;
#[cfg(feature = "python-extension")]
pub mod python;
mod traits;

use libc::{c_char, c_int, c_uchar, c_void, size_t};
pub use traits::{DoubleList, IntList, IntListOption};

#[repr(C)]
pub struct C_scalar {
    pub float_val: f64,
    pub int_val: i64,
}


    #[no_mangle] pub unsafe extern "C" fn ats_int(v: i64) -> *mut C_scalar { Box::into_raw(Box::new(C_scalar { float_val: v as f64, int_val: v })) }

    #[no_mangle] pub unsafe extern "C" fn ats_float(v: f64) -> *mut C_scalar { Box::into_raw(Box::new(C_scalar { float_val: v, int_val: v as i64 })) }

    #[no_mangle] pub unsafe extern "C" fn ats_to_int(arg: *mut C_scalar) -> i64 { if arg.is_null() { 0 } else { (*arg).int_val } }

    #[no_mangle] pub unsafe extern "C" fn ats_to_float(arg: *mut C_scalar) -> f64 { if arg.is_null() { 0.0 } else { (*arg).float_val } }

    #[no_mangle] pub unsafe extern "C" fn ats_to_string(arg: *mut C_scalar) -> *mut c_char { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ats_free(arg: *mut C_scalar) { if !arg.is_null() { drop(Box::from_raw(arg)); } }



#[repr(C)]
pub struct C_tensor {
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
    pub scalar_type: libc::c_int,
}

#[inline]
pub unsafe fn make_tensor(shape: Vec<i64>) -> *mut C_tensor {
    let num_elements: usize = shape.iter().map(|&x| (x.max(0)) as usize).product();
    let data = vec![0u8; num_elements.saturating_mul(4)];
    Box::into_raw(Box::new(C_tensor { shape, data, scalar_type: 6 }))
}

#[inline]
pub unsafe fn make_tensor_from_size(size_data: *const i64, size_len: libc::c_int, scalar_type: libc::c_int) -> *mut C_tensor {
    let shape = if !size_data.is_null() && size_len > 0 {
        std::slice::from_raw_parts(size_data, size_len as usize).to_vec()
    } else {
        vec![0]
    };
    let num_elements: usize = shape.iter().map(|&x| (x.max(0)) as usize).product();
    let elt_size = match scalar_type {
        4 => 8,
        7 => 8,
        _ => 4,
    };
    let data = vec![0u8; num_elements.saturating_mul(elt_size)];
    Box::into_raw(Box::new(C_tensor { shape, data, scalar_type }))
}

#[inline]
pub unsafe fn init_out_tensor(out__: *mut *mut C_tensor) {
    if !out__.is_null() {
        *out__ = make_tensor(vec![1, 64]);
    }
}


    #[no_mangle] pub unsafe extern "C" fn at_new_tensor() -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_shallow_clone(arg: *mut C_tensor) -> *mut C_tensor { if arg.is_null() { crate::make_tensor(vec![0]) } else { let t = &*arg; Box::into_raw(Box::new(C_tensor { shape: t.shape.clone(), data: t.data.clone(), scalar_type: t.scalar_type })) } }

    #[no_mangle] pub unsafe extern "C" fn at_copy_(_dst: *mut C_tensor, _src: *mut C_tensor) {}

    #[no_mangle] pub unsafe extern "C" fn at_data_ptr(arg: *mut C_tensor) -> *mut libc::c_void { if arg.is_null() { std::ptr::null_mut() } else { let t = &mut *arg; if t.data.is_empty() { std::ptr::null_mut() } else { t.data.as_mut_ptr() as *mut libc::c_void } } }

    #[no_mangle] pub unsafe extern "C" fn at_defined(arg: *mut C_tensor) -> libc::c_int { if arg.is_null() { 0 } else { 1 } }

    #[no_mangle] pub unsafe extern "C" fn at_is_sparse(arg: *mut C_tensor) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_is_mkldnn(arg: *mut C_tensor) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_is_contiguous(_args: *mut C_tensor) -> libc::c_int { 1 }

    #[no_mangle] pub unsafe extern "C" fn at_backward(arg: *mut C_tensor, keep_graph: c_int, create_graph: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn at_print(arg: *mut C_tensor) {}

    #[no_mangle] pub unsafe extern "C" fn at_to_string(arg: *mut C_tensor, line_size: c_int) -> *mut c_char { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn at_dim(arg: *mut C_tensor) -> libc::size_t { if arg.is_null() { 0 } else { let t = &*arg; t.shape.len() } }

    #[no_mangle] pub unsafe extern "C" fn at_get(arg: *mut C_tensor, index: libc::c_int) -> *mut C_tensor { if arg.is_null() { crate::make_tensor(vec![]) } else { let t = &*arg; if t.shape.is_empty() { crate::make_tensor(vec![]) } else { let shape = t.shape[1..].to_vec(); let numel: usize = shape.iter().map(|&x| (x.max(0)) as usize).product(); let elt_size = match t.scalar_type { 4 => 8, 7 => 8, _ => 4 }; let slice_bytes = numel * elt_size; let offset = (index as usize) * slice_bytes; let data = if offset + slice_bytes <= t.data.len() { t.data[offset..offset + slice_bytes].to_vec() } else { vec![0u8; slice_bytes] }; Box::into_raw(Box::new(C_tensor { shape, data, scalar_type: t.scalar_type })) } } }

    #[no_mangle] pub unsafe extern "C" fn at_requires_grad(arg: *mut C_tensor) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_shape(arg: *mut C_tensor, sz: *mut i64) { if !arg.is_null() && !sz.is_null() { let t = &*arg; for (i, &s) in t.shape.iter().enumerate() { *sz.add(i) = s; } } }

    #[no_mangle] pub unsafe extern "C" fn at_stride(arg: *mut C_tensor, sz: *mut i64) { if !arg.is_null() && !sz.is_null() { let t = &*arg; let mut stride = 1i64; for (i, &s) in t.shape.iter().rev().enumerate() { *sz.add(t.shape.len() - 1 - i) = stride; stride *= s.max(1); } } }

    #[no_mangle] pub unsafe extern "C" fn at_double_value_at_indexes(arg: *mut C_tensor, idx: *const i64, idx_len: libc::c_int) -> f64 {
        if !arg.is_null() {
            let t = &*arg;
            let mut elem_idx = 0usize;
            if !idx.is_null() && idx_len > 0 {
                let idx_slice = std::slice::from_raw_parts(idx, idx_len as usize);
                for (d, &i) in idx_slice.iter().enumerate() {
                    let stride = if d + 1 < t.shape.len() {
                        t.shape[d + 1..].iter().map(|&s| s.max(0) as usize).product()
                    } else {
                        1
                    };
                    elem_idx += (i.max(0) as usize) * stride;
                }
            }
            let byte_offset = elem_idx * 4;
            if byte_offset + 4 <= t.data.len() {
                let b = [t.data[byte_offset], t.data[byte_offset + 1], t.data[byte_offset + 2], t.data[byte_offset + 3]];
                f32::from_ne_bytes(b) as f64
            } else if t.data.len() >= 4 {
                let b = [t.data[0], t.data[1], t.data[2], t.data[3]];
                f32::from_ne_bytes(b) as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    #[no_mangle] pub unsafe extern "C" fn at_int64_value_at_indexes(arg: *mut C_tensor, idx: *const i64, idx_len: libc::c_int) -> i64 {
        if !arg.is_null() {
            let t = &*arg;
            let mut elem_idx = 0usize;
            if !idx.is_null() && idx_len > 0 {
                let idx_slice = std::slice::from_raw_parts(idx, idx_len as usize);
                for (d, &i) in idx_slice.iter().enumerate() {
                    let stride = if d + 1 < t.shape.len() {
                        t.shape[d + 1..].iter().map(|&s| s.max(0) as usize).product()
                    } else {
                        1
                    };
                    elem_idx += (i.max(0) as usize) * stride;
                }
            }
            let elt_size = match t.scalar_type { 4 => 8, 7 => 8, _ => 4 };
            let byte_offset = elem_idx * elt_size;
            if elt_size == 8 && byte_offset + 8 <= t.data.len() {
                let mut b = [0u8; 8];
                b.copy_from_slice(&t.data[byte_offset..byte_offset + 8]);
                i64::from_ne_bytes(b)
            } else if elt_size == 4 && byte_offset + 4 <= t.data.len() {
                let mut b = [0u8; 4];
                b.copy_from_slice(&t.data[byte_offset..byte_offset + 4]);
                i32::from_ne_bytes(b) as i64
            } else {
                1
            }
        } else {
            1
        }
    }

    #[no_mangle] pub unsafe extern "C" fn at_get_num_interop_threads() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_get_num_threads() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_set_num_interop_threads(n_threads: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn at_set_num_threads(n_threads: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn at_set_qengine(qengine: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn at_free(arg: *mut C_tensor) { if !arg.is_null() { drop(Box::from_raw(arg)); } }

    #[no_mangle] pub unsafe extern "C" fn at_run_backward(
        arg: *const *mut C_tensor,
        ntensors: c_int,
        inputs: *const *mut C_tensor,
        ninputs: c_int,
        outputs: *mut *mut C_tensor,
        keep_graph: c_int,
        create_graph: c_int,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_copy_data(arg: *mut C_tensor, vs: *const libc::c_void, numel: libc::size_t, elt_size_in_bytes: libc::size_t) { if !arg.is_null() && !vs.is_null() { let t = &*arg; let byte_count = numel * elt_size_in_bytes; let copy_bytes = byte_count.min(t.data.len()); std::ptr::copy_nonoverlapping(t.data.as_ptr(), vs as *mut u8, copy_bytes); } }

    #[no_mangle] pub unsafe extern "C" fn at_scalar_type(arg: *mut C_tensor) -> libc::c_int { if !arg.is_null() { let t = &*arg; t.scalar_type } else { 6 } }

    #[no_mangle] pub unsafe extern "C" fn at__amp_non_finite_check_and_unscale(
        t: *mut C_tensor,
        found_inf: *mut C_tensor,
        inf_scale: *mut C_tensor,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_autocast_clear_cache() {}

    #[no_mangle] pub unsafe extern "C" fn at_autocast_decrement_nesting() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_autocast_increment_nesting() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_autocast_is_enabled() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_autocast_set_enabled(b: c_int) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_device(arg: *mut C_tensor) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_tensor_of_data(data: *const libc::c_void, dims: *const i64, ndims: libc::size_t, element_size_in_bytes: libc::size_t, _type: libc::c_int) -> *mut C_tensor { let shape = if !dims.is_null() && ndims > 0 { std::slice::from_raw_parts(dims, ndims).to_vec() } else { vec![] }; let num_elements: usize = shape.iter().map(|&x| (x.max(0)) as usize).product(); let byte_len = num_elements * element_size_in_bytes; let data_vec = if !data.is_null() && byte_len > 0 { std::slice::from_raw_parts(data as *const u8, byte_len).to_vec() } else { vec![] }; Box::into_raw(Box::new(C_tensor { shape, data: data_vec, scalar_type: _type })) }

    #[no_mangle] pub unsafe extern "C" fn at_tensor_of_blob(
        vs: *const c_void,
        dims: *const i64,
        ndims: size_t,
        strides: *const i64,
        nstrides: size_t,
        kind: c_int,
        device: c_int,
    ) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_grad_set_enabled(b: c_int) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_save(arg: *mut C_tensor, filename: *const c_char) {}

    #[no_mangle] pub unsafe extern "C" fn at_save_to_stream(arg: *mut C_tensor, stream_ptr: *mut c_void) {}

    #[no_mangle] pub unsafe extern "C" fn at_load(filename: *const c_char) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_load_from_stream(stream_ptr: *mut c_void) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_save_multi(
        args: *const *mut C_tensor,
        names: *const *const c_char,
        n: c_int,
        filename: *const c_char,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_save_multi_to_stream(
        args: *const *mut C_tensor,
        names: *const *const c_char,
        n: c_int,
        stream_ptr: *mut c_void,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_loadz_callback(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_loadz_callback_with_device(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        device_id: c_int,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_load_callback(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_load_callback_with_device(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        device_id: c_int,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn at_load_from_stream_callback(
        stream_ptr: *mut c_void,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        enable_device_id: bool,
        device_id: c_int,
    ) {}


    #[no_mangle] pub unsafe extern "C" fn at_manual_seed(seed: i64) {}

    #[no_mangle] pub unsafe extern "C" fn at_set_graph_executor_optimize(b: bool) {}

    #[no_mangle] pub unsafe extern "C" fn at_context_has_openmp() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_mkl() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_lapack() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_mkldnn() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_magma() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_cuda() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_cudart() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_cusolver() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_hip() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_ipu() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_xla() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_lazy() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_has_mps() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn at_context_version_cudnn() -> i64 { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_context_version_cudart() -> i64 { 0 }



pub mod c_generated;


    #[no_mangle] pub unsafe extern "C" fn get_and_reset_last_err() -> *mut libc::c_char { std::ptr::null_mut() }



#[repr(C)]
pub struct C_optimizer {
    _private: [u8; 0],
}


    #[no_mangle] pub unsafe extern "C" fn ato_adam(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> *mut C_optimizer { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ato_adamw(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> *mut C_optimizer { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ato_rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: c_int,
    ) -> *mut C_optimizer { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ato_sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: c_int,
    ) -> *mut C_optimizer { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ato_add_parameters(arg: *mut C_optimizer, ts: *mut C_tensor, group: size_t) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_learning_rate(arg: *mut C_optimizer, lr: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_learning_rate_group(arg: *mut C_optimizer, group: size_t, lr: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_momentum(arg: *mut C_optimizer, momentum: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_momentum_group(arg: *mut C_optimizer, group: size_t, momentum: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_weight_decay(arg: *mut C_optimizer, weight_decay: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_set_weight_decay_group(arg: *mut C_optimizer, group: size_t, weight_decay: f64) {}

    #[no_mangle] pub unsafe extern "C" fn ato_zero_grad(arg: *mut C_optimizer) {}

    #[no_mangle] pub unsafe extern "C" fn ato_step(arg: *mut C_optimizer) {}

    #[no_mangle] pub unsafe extern "C" fn ato_free(arg: *mut C_optimizer) {}

    #[no_mangle] pub unsafe extern "C" fn at_save_image(arg: *mut C_tensor, filename: *const c_char) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn at_load_image(filename: *const c_char) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_load_image_from_memory(
        img_data: *const c_uchar,
        img_data_len: size_t,
    ) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn at_resize_image(arg: *mut C_tensor, out_w: c_int, out_h: c_int) -> *mut C_tensor { crate::make_tensor(vec![0]) }



#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
pub struct CIValue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CModule_ {
    _private: [u8; 0],
}


    // Constructors
    #[no_mangle] pub unsafe extern "C" fn ati_none() -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_bool(b: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_int(v: i64) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_double(v: f64) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_tensor(v: *mut C_tensor) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_string(s: *const c_char) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_tuple(v: *const *mut CIValue, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_generic_list(v: *const *mut CIValue, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_generic_dict(v: *const *mut CIValue, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_int_list(v: *const i64, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_double_list(v: *const f64, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_bool_list(v: *const c_char, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_string_list(v: *const *const c_char, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_tensor_list(v: *const *mut C_tensor, n: c_int) -> *mut CIValue { core::ptr::null_mut() }


    // Type query
    #[no_mangle] pub unsafe extern "C" fn ati_tag(arg: *mut CIValue) -> libc::c_int { 0 }


    // Getters
    #[no_mangle] pub unsafe extern "C" fn ati_to_int(arg: *mut CIValue) -> i64 { 0 }

    #[no_mangle] pub unsafe extern "C" fn ati_to_bool(arg: *mut CIValue) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn ati_to_double(arg: *mut CIValue) -> f64 { 0.0 }

    #[no_mangle] pub unsafe extern "C" fn ati_to_tensor(arg: *mut CIValue) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn ati_length(arg: *mut CIValue) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn ati_tuple_length(arg: *mut CIValue) -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn ati_to_tuple(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_generic_list(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_generic_dict(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_int_list(arg: *mut CIValue, outputs: *mut i64, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_double_list(arg: *mut CIValue, outputs: *mut f64, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_bool_list(arg: *mut CIValue, outputs: *mut c_char, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_tensor_list(arg: *mut CIValue, outputs: *mut *mut C_tensor, n: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn ati_to_string(arg: *mut CIValue) -> *mut c_char { core::ptr::null_mut() }


    #[no_mangle] pub unsafe extern "C" fn ati_clone(arg: *mut CIValue) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn ati_free(arg: *mut CIValue) {}


    #[no_mangle] pub unsafe extern "C" fn ati_object_method_(
        arg: *mut CIValue,
        method_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue { core::ptr::null_mut() }


    #[no_mangle] pub unsafe extern "C" fn ati_object_getattr_(arg: *mut CIValue, attr_name: *const c_char) -> *mut CIValue { core::ptr::null_mut() }


    #[no_mangle] pub unsafe extern "C" fn atm_load(filename: *const c_char) -> *mut CModule_ { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_load_on_device(filename: *const c_char, device: c_int) -> *mut CModule_ { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_load_str(data: *const c_char, sz: size_t) -> *mut CModule_ { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_load_str_on_device(data: *const c_char, sz: size_t, device: c_int) -> *mut CModule_ { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_forward(m: *mut CModule_, args: *const *mut C_tensor, n: c_int) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn atm_forward_(m: *mut CModule_, args: *const *mut CIValue, n: c_int) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_method(
        m: *mut CModule_,
        method_name: *const c_char,
        args: *const *mut C_tensor,
        n: c_int,
    ) -> *mut C_tensor { crate::make_tensor(vec![0]) }

    #[no_mangle] pub unsafe extern "C" fn atm_method_(
        m: *mut CModule_,
        method_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_create_class_(
        m: *mut CModule_,
        clz_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_eval(m: *mut CModule_) {}

    #[no_mangle] pub unsafe extern "C" fn atm_train(m: *mut CModule_) {}

    #[no_mangle] pub unsafe extern "C" fn atm_free(m: *mut CModule_) {}

    #[no_mangle] pub unsafe extern "C" fn atm_to(m: *mut CModule_, device: c_int, kind: c_int, non_blocking: bool) {}

    #[no_mangle] pub unsafe extern "C" fn atm_save(m: *mut CModule_, filename: *const c_char) {}

    #[no_mangle] pub unsafe extern "C" fn atm_get_profiling_mode() -> libc::c_int { 0 }

    #[no_mangle] pub unsafe extern "C" fn atm_set_profiling_mode(profiling_mode: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn atm_fuser_cuda_set_enabled(enabled: bool) {}

    #[no_mangle] pub unsafe extern "C" fn atm_fuser_cuda_is_enabled() -> bool { false }

    #[no_mangle] pub unsafe extern "C" fn atm_named_parameters(
        m: *mut CModule_,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    ) {}

    #[no_mangle] pub unsafe extern "C" fn atm_create_for_tracing(
        modl_name: *const c_char,
        inputs: *const *mut C_tensor,
        ninputs: c_int,
    ) -> *mut CModule_ { core::ptr::null_mut() }

    #[no_mangle] pub unsafe extern "C" fn atm_end_tracing(
        m: *mut CModule_,
        fn_name: *const c_char,
        outputs: *const *mut C_tensor,
        noutputs: c_int,
    ) {}

    #[no_mangle] pub unsafe extern "C" fn atm_set_tensor_expr_fuser_enabled(enabled: c_int) {}

    #[no_mangle] pub unsafe extern "C" fn atm_get_tensor_expr_fuser_enabled() -> bool { false }


