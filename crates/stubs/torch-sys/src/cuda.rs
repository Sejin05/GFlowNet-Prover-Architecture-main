#![allow(warnings)]
use libc::c_int;

#[no_mangle] pub unsafe extern "C" fn atc_cuda_device_count() -> c_int { 0 }
#[no_mangle] pub unsafe extern "C" fn atc_cuda_is_available() -> c_int { 0 }
#[no_mangle] pub unsafe extern "C" fn atc_cudnn_is_available() -> c_int { 0 }
#[no_mangle] pub unsafe extern "C" fn atc_manual_seed(_seed: u64) {}
#[no_mangle] pub unsafe extern "C" fn atc_manual_seed_all(_seed: u64) {}
#[no_mangle] pub unsafe extern "C" fn atc_synchronize(_device_index: i64) {}
#[no_mangle] pub unsafe extern "C" fn atc_user_enabled_cudnn() -> c_int { 0 }
#[no_mangle] pub unsafe extern "C" fn atc_set_user_enabled_cudnn(_b: c_int) {}
#[no_mangle] pub unsafe extern "C" fn atc_set_benchmark_cudnn(_b: c_int) {}
