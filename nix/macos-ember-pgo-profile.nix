{
  pkgs,
  nixpkgs,
  rust-overlay,
  lib,
  arch,
}:

# Builds an instrumented macOS ember for the target architecture, runs a
# deterministic fixed-depth bench workload (natively for arm64, under
# Rosetta 2 for amd64), and installs the merged LLVM profile. The same-arch
# profile is reused by the macOS release package of that architecture.
let
  target =
    {
      amd64 = "x86_64-apple-darwin";
      arm64 = "aarch64-apple-darwin";
    }
    .${arch} or (throw "unsupported macOS PGO profile architecture: ${arch}");
  targetSystem =
    {
      amd64 = "x86_64-darwin";
      arm64 = "aarch64-darwin";
    }
    .${arch};
  targetCpu =
    {
      amd64 = "x86-64";
      arm64 = "apple-m1";
    }
    .${arch};
  nativeSystem = pkgs.stdenv.hostPlatform.system;
  # Nix names the arm64 Darwin host "arm64-apple-darwin", while Rust
  # installs host tools under "aarch64-apple-darwin". The host may also
  # differ from the instrumented target when training under Rosetta.
  toolchainHost =
    {
      "x86_64-darwin" = "x86_64-apple-darwin";
      "aarch64-darwin" = "aarch64-apple-darwin";
    }
    .${nativeSystem};
  profileSystem =
    {
      amd64 = "x86_64-darwin";
      arm64 = "aarch64-darwin";
    }
    .${arch};
  # Rosetta 2 runs the cross-arch instrumented binary inside the sandbox.
  emulator = if nativeSystem == profileSystem then "" else "/usr/bin/arch -x86_64 ";
  targetPackages = import nixpkgs (
    {
      localSystem = pkgs.stdenv.hostPlatform.system;
      overlays = [ (import rust-overlay) ];
    }
    // lib.optionalAttrs (pkgs.stdenv.hostPlatform.system != targetSystem) {
      crossSystem = {
        config = target;
      };
    }
  );
  targetCc = targetPackages.stdenv.cc;
  linker = "${targetCc}/bin/${targetCc.targetPrefix}cc";
  cargoTargetEnv = lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] target);
  rustToolchainConfig = (builtins.fromTOML (builtins.readFile ../rust-toolchain.toml)).toolchain;
  rustToolchain = pkgs.rust-bin.fromRustupToolchain (
    rustToolchainConfig
    // {
      targets = [ target ];
      components = rustToolchainConfig.components ++ [ "llvm-tools-preview" ];
    }
  );
  # Raw profile format follows rustc's bundled LLVM, which can differ from nixpkgs LLVM.
  llvmProfdata = "${rustToolchain}/lib/rustlib/${toolchainHost}/bin/llvm-profdata";
  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
  version = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).package.version;
in
rustPlatform.buildRustPackage {
  pname = "ember-macos-pgo-profile-${arch}";
  inherit version;

  src = import ./ember-source.nix { inherit lib pkgs; };
  cargoLock = import ./ember-cargo-lock.nix;

  nativeBuildInputs = [
    targetCc
  ];

  buildPhase = ''
    runHook preBuild
    test -x ${llvmProfdata} || {
      echo "Rust llvm-profdata is missing for host ${toolchainHost}" >&2
      exit 1
    }
    export CARGO_TARGET_${cargoTargetEnv}_LINKER="${linker}"
    export MACOSX_DEPLOYMENT_TARGET=11.0
    export RUSTFLAGS="-C target-cpu=${targetCpu} -Cprofile-generate=$PWD/profraw"
    mkdir -p profraw
    cargo build --frozen --release --bin ember --target ${target}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    printf 'bench depth 12\nbench depth 14\nquit\n' \
      | ${emulator}target/${target}/release/ember >/dev/null
    ${llvmProfdata} merge \
      -o merged.profdata profraw/*.profraw
    test -s merged.profdata || {
      echo "PGO profile merge produced no data" >&2
      exit 1
    }
    mkdir -p "$out"
    cp merged.profdata "$out/merged.profdata"
    runHook postInstall
  '';

  doCheck = false;
  dontFixup = true;

  passthru = {
    inherit arch target;
    workload = "bench depth 12; bench depth 14";
  };

  meta = {
    description = "LLVM PGO profile for Ember macOS ${arch} release builds";
    mainProgram = "ember";
  };
}
