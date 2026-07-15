use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NnueBackendKind {
    Scalar,
    Simd128,
    Simd256,
    Simd512,
    X86Avx512,
}

impl NnueBackendKind {
    pub fn name(self) -> &'static str {
        match self {
            NnueBackendKind::Scalar => "scalar",
            NnueBackendKind::Simd128 => "simd128",
            NnueBackendKind::Simd256 => "simd256",
            NnueBackendKind::Simd512 => "simd512",
            NnueBackendKind::X86Avx512 => "x86-avx512",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SearchBackendKind {
    Scalar,
    X86V3,
    Aarch64Simd128,
    Aarch64Simd256,
    Aarch64Simd512,
    X86Avx512,
}

impl SearchBackendKind {
    pub fn name(self) -> &'static str {
        match self {
            SearchBackendKind::Scalar => "scalar",
            SearchBackendKind::X86V3 => "x86-v3",
            SearchBackendKind::Aarch64Simd128 => "aarch64-simd128",
            SearchBackendKind::Aarch64Simd256 => "aarch64-simd256",
            SearchBackendKind::Aarch64Simd512 => "aarch64-simd512",
            SearchBackendKind::X86Avx512 => "x86-avx512",
        }
    }

    pub fn nnue_backend(self) -> NnueBackendKind {
        match self {
            SearchBackendKind::Scalar => NnueBackendKind::Scalar,
            SearchBackendKind::Aarch64Simd128 => NnueBackendKind::Simd128,
            SearchBackendKind::X86V3 | SearchBackendKind::Aarch64Simd256 => {
                NnueBackendKind::Simd256
            }
            SearchBackendKind::Aarch64Simd512 => NnueBackendKind::Simd512,
            SearchBackendKind::X86Avx512 => NnueBackendKind::X86Avx512,
        }
    }
}

pub fn parse_search_backend_name(value: &str) -> Option<SearchBackendKind> {
    match normalize_backend_name(value).as_str() {
        "scalar" | "portable" => Some(SearchBackendKind::Scalar),
        "x86-v3" | "x86-64-v3" | "x86_64-v3" | "v3" | "avx2" => Some(SearchBackendKind::X86V3),
        "aarch64-simd128" | "arm64-simd128" | "arm-simd128" | "neon128" => {
            Some(SearchBackendKind::Aarch64Simd128)
        }
        "aarch64-simd" | "arm64-simd" | "arm-simd" | "neon" | "aarch64-simd256"
        | "arm64-simd256" | "arm-simd256" | "neon256" => Some(SearchBackendKind::Aarch64Simd256),
        "aarch64-simd512" | "arm64-simd512" | "arm-simd512" | "neon512" => {
            Some(SearchBackendKind::Aarch64Simd512)
        }
        "x86-avx512" | "avx512" | "x86-v4" | "x86-64-v4" | "v4" => {
            Some(SearchBackendKind::X86Avx512)
        }
        "simd" => Some(default_search_backend()),
        "simd256" => {
            if x86_v3_available() {
                Some(SearchBackendKind::X86V3)
            } else if aarch64_simd_available() {
                Some(SearchBackendKind::Aarch64Simd256)
            } else {
                Some(SearchBackendKind::Scalar)
            }
        }
        "" | "auto" => None,
        _ => None,
    }
}

pub fn normalize_backend_name(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

pub fn default_search_backend() -> SearchBackendKind {
    static DEFAULT_SEARCH_BACKEND: OnceLock<SearchBackendKind> = OnceLock::new();
    *DEFAULT_SEARCH_BACKEND.get_or_init(|| {
        if x86_avx512_available() {
            SearchBackendKind::X86Avx512
        } else if x86_v3_available() {
            SearchBackendKind::X86V3
        } else {
            SearchBackendKind::Scalar
        }
    })
}

pub fn compiled_search_backends() -> Vec<SearchBackendKind> {
    let mut backends = vec![SearchBackendKind::Scalar];

    #[cfg(target_arch = "x86_64")]
    {
        backends.push(SearchBackendKind::X86V3);
        backends.push(SearchBackendKind::X86Avx512);
    }
    #[cfg(target_arch = "aarch64")]
    {
        backends.push(SearchBackendKind::Aarch64Simd128);
        backends.push(SearchBackendKind::Aarch64Simd256);
        backends.push(SearchBackendKind::Aarch64Simd512);
    }

    backends
}

pub fn available_search_backends() -> Vec<SearchBackendKind> {
    let mut backends = vec![SearchBackendKind::Scalar];

    if x86_v3_available() {
        backends.push(SearchBackendKind::X86V3);
    }
    if aarch64_simd_available() {
        backends.push(SearchBackendKind::Aarch64Simd128);
        backends.push(SearchBackendKind::Aarch64Simd256);
        backends.push(SearchBackendKind::Aarch64Simd512);
    }
    if x86_avx512_available() {
        backends.push(SearchBackendKind::X86Avx512);
    }

    backends
}

pub fn search_backend_available(backend: SearchBackendKind) -> bool {
    match backend {
        SearchBackendKind::Scalar => true,
        SearchBackendKind::X86V3 => x86_v3_available(),
        SearchBackendKind::Aarch64Simd128
        | SearchBackendKind::Aarch64Simd256
        | SearchBackendKind::Aarch64Simd512 => aarch64_simd_available(),
        SearchBackendKind::X86Avx512 => x86_avx512_available(),
    }
}

pub fn available_nnue_backends() -> Vec<NnueBackendKind> {
    let mut backends = vec![NnueBackendKind::Scalar];

    if aarch64_simd_available() {
        backends.push(NnueBackendKind::Simd128);
    }
    if x86_v3_available() || aarch64_simd_available() {
        backends.push(NnueBackendKind::Simd256);
    }
    if aarch64_simd_available() {
        backends.push(NnueBackendKind::Simd512);
    }
    if x86_avx512_available() {
        backends.push(NnueBackendKind::X86Avx512);
    }

    backends
}

pub fn nnue_backend_available(backend: NnueBackendKind) -> bool {
    match backend {
        NnueBackendKind::Scalar => true,
        NnueBackendKind::Simd128 => aarch64_simd_available(),
        NnueBackendKind::Simd256 => x86_v3_available() || aarch64_simd_available(),
        NnueBackendKind::Simd512 => aarch64_simd_available(),
        NnueBackendKind::X86Avx512 => x86_avx512_available(),
    }
}

#[inline]
pub fn x86_v3_available() -> bool {
    static X86_V3_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *X86_V3_AVAILABLE.get_or_init(x86_v3_available_impl)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_v3_available_impl() -> bool {
    std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn x86_v3_available_impl() -> bool {
    false
}

#[inline]
pub fn x86_avx512_available() -> bool {
    static X86_AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *X86_AVX512_AVAILABLE.get_or_init(x86_avx512_available_impl)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_avx512_available_impl() -> bool {
    x86_v3_available()
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn x86_avx512_available_impl() -> bool {
    false
}

#[inline]
pub fn aarch64_simd_available() -> bool {
    static AARCH64_SIMD_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AARCH64_SIMD_AVAILABLE.get_or_init(aarch64_simd_available_impl)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn aarch64_simd_available_impl() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn aarch64_simd_available_impl() -> bool {
    false
}
