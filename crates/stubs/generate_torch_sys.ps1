$ErrorActionPreference = "Stop"

$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $HOME ".cargo" }
$torchSysCandidate = Get-ChildItem -Path (Join-Path $cargoHome "registry\src") -Filter "torch-sys-0.18.1" -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $torchSysCandidate) {
    throw "Could not locate torch-sys-0.18.1 under $cargoHome\registry\src"
}
$torchSysSrc = $torchSysCandidate.FullName
$outDir = "crates\stubs\torch-sys"

New-Item -ItemType Directory -Force -Path "$outDir\src" | Out-Null

# 1. Cargo.toml
@"
[package]
name = "torch-sys"
version = "0.18.1"
edition = "2021"
license = "MIT OR Apache-2.0"

[dependencies]
libc = "0.2"

[features]
default = []
doc-only = []
python-extension = []
ureq = []
"@ | Set-Content "$outDir\Cargo.toml" -Encoding utf8

# 2. Copy traits.rs and io.rs directly
Copy-Item "$torchSysSrc\src\traits.rs" "$outDir\src\traits.rs"
Copy-Item "$torchSysSrc\src\io.rs" "$outDir\src\io.rs"

# 3. Generate cuda.rs
@"
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
"@ | Set-Content "$outDir\src\cuda.rs" -Encoding utf8

function Transform-FnBlock {
    param([string]$code)

    $regex = '(?s)pub fn (at\w+|get_and_reset_last_err)\((.*?)\)(?:\s*->\s*([^;]+))?;'

    $result = [regex]::Replace($code, $regex, {
        param($m)
        $name = $m.Groups[1].Value
        $params = $m.Groups[2].Value
        $ret = $m.Groups[3].Value.Trim()

        # Specific overrides
        if ($name -eq "ats_float") {
            return "#[no_mangle] pub unsafe extern `"C`" fn ats_float(v: f64) -> *mut C_scalar { Box::into_raw(Box::new(C_scalar { float_val: v, int_val: v as i64 })) }`n"
        }
        if ($name -eq "ats_int") {
            return "#[no_mangle] pub unsafe extern `"C`" fn ats_int(v: i64) -> *mut C_scalar { Box::into_raw(Box::new(C_scalar { float_val: v as f64, int_val: v })) }`n"
        }
        if ($name -eq "ats_to_float") {
            return "#[no_mangle] pub unsafe extern `"C`" fn ats_to_float(arg: *mut C_scalar) -> f64 { if arg.is_null() { 0.0 } else { (*arg).float_val } }`n"
        }
        if ($name -eq "ats_to_int") {
            return "#[no_mangle] pub unsafe extern `"C`" fn ats_to_int(arg: *mut C_scalar) -> i64 { if arg.is_null() { 0 } else { (*arg).int_val } }`n"
        }
        if ($name -eq "ats_free") {
            return "#[no_mangle] pub unsafe extern `"C`" fn ats_free(arg: *mut C_scalar) { if !arg.is_null() { drop(Box::from_raw(arg)); } }`n"
        }
        if ($name -eq "at_scalar_type") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_scalar_type(arg: *mut C_tensor) -> libc::c_int { if !arg.is_null() { let t = &*arg; t.scalar_type } else { 6 } }`n"
        }
        if ($name -eq "at_defined") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_defined(arg: *mut C_tensor) -> libc::c_int { if arg.is_null() { 0 } else { 1 } }`n"
        }
        if ($name -eq "at_dim") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_dim(arg: *mut C_tensor) -> libc::size_t { if arg.is_null() { 0 } else { let t = &*arg; t.shape.len() } }`n"
        }
        if ($name -eq "at_shape") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_shape(arg: *mut C_tensor, sz: *mut i64) { if !arg.is_null() && !sz.is_null() { let t = &*arg; for (i, &s) in t.shape.iter().enumerate() { *sz.add(i) = s; } } }`n"
        }
        if ($name -eq "at_stride") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_stride(arg: *mut C_tensor, sz: *mut i64) { if !arg.is_null() && !sz.is_null() { let t = &*arg; let mut stride = 1i64; for (i, &s) in t.shape.iter().rev().enumerate() { *sz.add(t.shape.len() - 1 - i) = stride; stride *= s.max(1); } } }`n"
        }
        if ($name -eq "at_free") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_free(arg: *mut C_tensor) { if !arg.is_null() { drop(Box::from_raw(arg)); } }`n"
        }
        if ($name -eq "at_new_tensor") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_new_tensor() -> *mut C_tensor { crate::make_tensor(vec![0]) }`n"
        }
        if ($name -eq "at_shallow_clone") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_shallow_clone(arg: *mut C_tensor) -> *mut C_tensor { if arg.is_null() { crate::make_tensor(vec![0]) } else { let t = &*arg; Box::into_raw(Box::new(C_tensor { shape: t.shape.clone(), data: t.data.clone(), scalar_type: t.scalar_type })) } }`n"
        }
        if ($name -eq "at_data_ptr") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_data_ptr(arg: *mut C_tensor) -> *mut libc::c_void { if arg.is_null() { std::ptr::null_mut() } else { let t = &mut *arg; if t.data.is_empty() { std::ptr::null_mut() } else { t.data.as_mut_ptr() as *mut libc::c_void } } }`n"
        }
        if ($name -eq "at_tensor_of_data") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_tensor_of_data(data: *const libc::c_void, dims: *const i64, ndims: libc::size_t, element_size_in_bytes: libc::size_t, _type: libc::c_int) -> *mut C_tensor { let shape = if !dims.is_null() && ndims > 0 { std::slice::from_raw_parts(dims, ndims).to_vec() } else { vec![] }; let num_elements: usize = shape.iter().map(|&x| (x.max(0)) as usize).product(); let byte_len = num_elements * element_size_in_bytes; let data_vec = if !data.is_null() && byte_len > 0 { std::slice::from_raw_parts(data as *const u8, byte_len).to_vec() } else { vec![] }; Box::into_raw(Box::new(C_tensor { shape, data: data_vec, scalar_type: _type })) }`n"
        }
        if ($name -eq "at_copy_data") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_copy_data(arg: *mut C_tensor, vs: *const libc::c_void, numel: libc::size_t, elt_size_in_bytes: libc::size_t) { if !arg.is_null() && !vs.is_null() { let t = &*arg; let byte_count = numel * elt_size_in_bytes; let copy_bytes = byte_count.min(t.data.len()); std::ptr::copy_nonoverlapping(t.data.as_ptr(), vs as *mut u8, copy_bytes); } }`n"
        }
        if ($name -eq "at_get") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_get(arg: *mut C_tensor, index: libc::c_int) -> *mut C_tensor { if arg.is_null() { crate::make_tensor(vec![]) } else { let t = &*arg; if t.shape.is_empty() { crate::make_tensor(vec![]) } else { let shape = t.shape[1..].to_vec(); let numel: usize = shape.iter().map(|&x| (x.max(0)) as usize).product(); let elt_size = match t.scalar_type { 4 => 8, 7 => 8, _ => 4 }; let slice_bytes = numel * elt_size; let offset = (index as usize) * slice_bytes; let data = if offset + slice_bytes <= t.data.len() { t.data[offset..offset + slice_bytes].to_vec() } else { vec![0u8; slice_bytes] }; Box::into_raw(Box::new(C_tensor { shape, data, scalar_type: t.scalar_type })) } } }`n"
        }
        if ($name -eq "at_copy_") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_copy_(_dst: *mut C_tensor, _src: *mut C_tensor) {}`n"
        }
        if ($name -eq "get_and_reset_last_err") {
            return "#[no_mangle] pub unsafe extern `"C`" fn get_and_reset_last_err() -> *mut libc::c_char { std::ptr::null_mut() }`n"
        }
        if ($name -eq "at_int64_value_at_indexes" -or $name -eq "at_int64_value") {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name(arg: *mut C_tensor, _idx: *const i64, _idx_len: libc::c_int) -> i64 { if !arg.is_null() { let t = &*arg; if t.data.len() >= 8 { let mut b = [0u8; 8]; b.copy_from_slice(&t.data[..8]); i64::from_ne_bytes(b) } else { 1 } } else { 1 } }`n"
        }
        if ($name -eq "at_double_value_at_indexes" -or $name -eq "at_double_value") {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name(arg: *mut C_tensor, _idx: *const i64, _idx_len: libc::c_int) -> f64 { if !arg.is_null() { let t = &*arg; if t.data.len() >= 4 { let b = [t.data[0], t.data[1], t.data[2], t.data[3]]; f32::from_ne_bytes(b) as f64 } else { 0.0 } } else { 0.0 } }`n"
        }
        if ($name -eq "at_is_contiguous") {
            return "#[no_mangle] pub unsafe extern `"C`" fn at_is_contiguous(_args: *mut C_tensor) -> libc::c_int { 1 }`n"
        }
        if ($name -eq "atg_zeros" -or $name -eq "atg_ones" -or $name -eq "atg_empty") {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) { if !out__.is_null() { *out__ = crate::make_tensor_from_size(size_data, size_len, options_kind); } }`n"
        }
        if ($name -eq "atg_view") {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) { if !out__.is_null() { let st = if !self_.is_null() { (*self_).scalar_type } else { 6 }; let t = crate::make_tensor_from_size(size_data, size_len, st); if !self_.is_null() { (*t).data = (*self_).data.clone(); } *out__ = t; } }`n"
        }
        if ($name -eq "atg_reshape") {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) { if !out__.is_null() { let st = if !self_.is_null() { (*self_).scalar_type } else { 6 }; let t = crate::make_tensor_from_size(shape_data, shape_len, st); if !self_.is_null() { (*t).data = (*self_).data.clone(); } *out__ = t; } }`n"
        }
        if ($name -eq "atg_transpose") {
            return "#[no_mangle] pub unsafe extern `"C`" fn atg_transpose(out__: *mut *mut C_tensor, self_: *mut C_tensor, dim0: i64, dim1: i64) { if !out__.is_null() { if !self_.is_null() { let t = &*self_; let mut shape = t.shape.clone(); let d0 = dim0 as usize; let d1 = dim1 as usize; if d0 < shape.len() && d1 < shape.len() { shape.swap(d0, d1); } *out__ = Box::into_raw(Box::new(C_tensor { shape, data: t.data.clone(), scalar_type: t.scalar_type })); } else { *out__ = crate::make_tensor(vec![0]); } } }`n"
        }
        if ($name -eq "atg_all") {
            return "#[no_mangle] pub unsafe extern `"C`" fn atg_all(out__: *mut *mut C_tensor, _self_: *mut C_tensor) { if !out__.is_null() { *out__ = Box::into_raw(Box::new(C_tensor { shape: vec![], data: 1i64.to_ne_bytes().to_vec(), scalar_type: 4 })); } }`n"
        }
        if ($name -eq "atg_fill_") {
            return "#[no_mangle] pub unsafe extern `"C`" fn atg_fill_(out__: *mut *mut C_tensor, self_: *mut C_tensor, value_: *mut C_scalar) { if !self_.is_null() && !value_.is_null() { let t = &mut *self_; let f = crate::ats_to_float(value_) as f32; let bytes = f.to_ne_bytes(); for chunk in t.data.chunks_exact_mut(4) { chunk.copy_from_slice(&bytes); } } if !out__.is_null() { *out__ = self_; } }`n"
        }
        if ($params -like "*out__:*") {
            if ($params -like "*self_: *mut C_tensor*") {
                return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) { if !out__.is_null() { *out__ = if !self_.is_null() { let t = &*self_; Box::into_raw(Box::new(C_tensor { shape: t.shape.clone(), data: t.data.clone(), scalar_type: t.scalar_type })) } else { crate::make_tensor(vec![0]) }; } }`n"
            } else {
                return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) { crate::init_out_tensor(out__); }`n"
            }
        }

        # Generic return handlers
        if ($ret) {
            if ($ret -eq "bool") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> bool { false }`n" }
            if ($ret -eq "i64") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> i64 { 0 }`n" }
            if ($ret -eq "libc::c_int" -or $ret -eq "c_int") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> libc::c_int { 0 }`n" }
            if ($ret -eq "libc::size_t" -or $ret -eq "size_t") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> libc::size_t { 0 }`n" }
            if ($ret -eq "f64") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> f64 { 0.0 }`n" }
            if ($ret -eq "*mut C_tensor") { return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> *mut C_tensor { crate::make_tensor(vec![0]) }`n" }
            return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) -> $ret { core::ptr::null_mut() }`n"
        } else {
            return "#[no_mangle] pub unsafe extern `"C`" fn $name($params) {}`n"
        }
    })
    return $result
}

# 4. Generate lib.rs
Write-Host "Generating lib.rs..."
$libRaw = [System.IO.File]::ReadAllText("$torchSysSrc\src\lib.rs")

$libRaw = $libRaw -replace '(?s)#\[repr\(C\)\]\s*pub struct C_scalar\s*\{[^}]*\}', @'
#[repr(C)]
pub struct C_scalar {
    pub float_val: f64,
    pub int_val: i64,
}
'@

$libRaw = $libRaw -replace '(?s)#\[repr\(C\)\]\s*pub struct C_tensor\s*\{[^}]*\}', @'
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
'@

# Replace each extern "C" { ... } block in lib.rs
$transformedLib = [regex]::Replace($libRaw, '(?s)extern "C"\s*\{(?<body>.*?)\}', {
    param($m)
    Transform-FnBlock $m.Groups['body'].Value
})

$libFinal = "#![allow(warnings)]`npub use c_generated::*;`n" + $transformedLib
[System.IO.File]::WriteAllText("$outDir\src\lib.rs", $libFinal, [System.Text.Encoding]::UTF8)
Write-Host "lib.rs generated."

# 5. Generate c_generated.rs
Write-Host "Generating c_generated.rs..."
$cGenRaw = [System.IO.File]::ReadAllText("$torchSysSrc\src\c_generated.rs")

$transformedCGen = [regex]::Replace($cGenRaw, '(?s)extern "C"\s*\{(?<body>.*?)\}', {
    param($m)
    Transform-FnBlock $m.Groups['body'].Value
})

$cGenFinal = "#![allow(warnings)]`n/* THIS FILE IS AUTOMATICALLY GENERATED FOR PURE RUST TESTING */`n" + $transformedCGen
[System.IO.File]::WriteAllText("$outDir\src\c_generated.rs", $cGenFinal, [System.Text.Encoding]::UTF8)
Write-Host "c_generated.rs successfully generated."
