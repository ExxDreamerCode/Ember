#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NnueBackendKind {
    Scalar,
    Simd256,
    X86Avx512,
}

impl NnueBackendKind {
    pub fn name(self) -> &'static str {
        match self {
            NnueBackendKind::Scalar => "scalar",
            NnueBackendKind::Simd256 => "simd256",
            NnueBackendKind::X86Avx512 => "x86-avx512",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SearchBackendKind {
    Scalar,
    X86V3,
    Aarch64Simd,
    X86Avx512,
}

impl SearchBackendKind {
    pub fn name(self) -> &'static str {
        match self {
            SearchBackendKind::Scalar => "scalar",
            SearchBackendKind::X86V3 => "x86-v3",
            SearchBackendKind::Aarch64Simd => "aarch64-simd",
            SearchBackendKind::X86Avx512 => "x86-avx512",
        }
    }

    pub fn nnue_backend(self) -> NnueBackendKind {
        match self {
            SearchBackendKind::Scalar => NnueBackendKind::Scalar,
            SearchBackendKind::X86V3 | SearchBackendKind::Aarch64Simd => NnueBackendKind::Simd256,
            SearchBackendKind::X86Avx512 => NnueBackendKind::X86Avx512,
        }
    }
}

pub fn parse_search_backend_name(value: &str) -> Option<SearchBackendKind> {
    match normalize_backend_name(value).as_str() {
        "scalar" | "portable" => Some(SearchBackendKind::Scalar),
        "x86-v3" | "x86-64-v3" | "x86_64-v3" | "v3" | "avx2" => Some(SearchBackendKind::X86V3),
        "aarch64-simd" | "arm64-simd" | "arm-simd" | "neon" => Some(SearchBackendKind::Aarch64Simd),
        "x86-avx512" | "avx512" | "x86-v4" | "x86-64-v4" | "v4" => {
            Some(SearchBackendKind::X86Avx512)
        }
        "simd" | "simd256" => {
            if x86_v3_available() {
                Some(SearchBackendKind::X86V3)
            } else if aarch64_simd_available() {
                Some(SearchBackendKind::Aarch64Simd)
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
    if x86_v3_available() {
        SearchBackendKind::X86V3
    } else {
        SearchBackendKind::Scalar
    }
}

pub fn available_search_backends() -> Vec<SearchBackendKind> {
    let mut backends = vec![SearchBackendKind::Scalar];

    if x86_v3_available() {
        backends.push(SearchBackendKind::X86V3);
    }
    if aarch64_simd_available() {
        backends.push(SearchBackendKind::Aarch64Simd);
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
        SearchBackendKind::Aarch64Simd => aarch64_simd_available(),
        SearchBackendKind::X86Avx512 => x86_avx512_available(),
    }
}

pub fn available_nnue_backends() -> Vec<NnueBackendKind> {
    let mut backends = vec![NnueBackendKind::Scalar];

    if x86_v3_available() || aarch64_simd_available() {
        backends.push(NnueBackendKind::Simd256);
    }
    if x86_avx512_available() {
        backends.push(NnueBackendKind::X86Avx512);
    }

    backends
}

pub fn nnue_backend_available(backend: NnueBackendKind) -> bool {
    match backend {
        NnueBackendKind::Scalar => true,
        NnueBackendKind::Simd256 => x86_v3_available() || aarch64_simd_available(),
        NnueBackendKind::X86Avx512 => x86_avx512_available(),
    }
}

#[inline]
pub fn x86_v3_available() -> bool {
    x86_v3_available_impl()
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
    x86_avx512_available_impl()
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
    aarch64_simd_available_impl()
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
