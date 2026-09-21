{
  pkgs,
  lib,
  arch,
}:

# Builds an instrumented static Linux ember, runs a deterministic fixed-depth
# bench workload, and installs the merged LLVM profile. PGO data is
# function-level execution counting over target-independent IR, so the
# same-arch profile is reused by the Linux and Windows release packages.
# macOS collects its own target-specific profile separately.
#
# amd64 runs natively on x86_64 builders. Generic arm64 always trains under
# QEMU with and without DOTPROD, including on native builders. This keeps
# profile coverage independent of the builder's CPU. These are execution
# counts, not performance measurements.
let
  target =
    {
      amd64 = "x86_64-unknown-linux-musl";
      arm64 = "aarch64-unknown-linux-musl";
    }
    .${arch} or (throw "unsupported PGO profile architecture: ${arch}");
  targetCpu =
    {
      amd64 = "x86-64-v3";
      arm64 = "generic";
    }
    .${arch};
  nativeSystem = pkgs.stdenv.hostPlatform.system;
  profileSystem =
    {
      amd64 = "x86_64-linux";
      arm64 = "aarch64-linux";
    }
    .${arch};
  emulator =
    if nativeSystem == profileSystem then
      ""
    else
      "${pkgs.qemu-user}/bin/qemu-${if arch == "arm64" then "aarch64" else "x86_64"} ";
  crossPackages =
    {
      amd64 = pkgs.pkgsCross.musl64;
      arm64 = pkgs.pkgsCross.aarch64-multiplatform-musl;
    }
    .${arch};
  crossCc = crossPackages.stdenv.cc;
  linker = "${crossCc}/bin/${crossCc.targetPrefix}cc";
  archiver = "${crossCc}/bin/${crossCc.targetPrefix}ar";
  cargoTargetEnv = lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] target);
  rustToolchainConfig = (builtins.fromTOML (builtins.readFile ../rust-toolchain.toml)).toolchain;
  rustToolchain = pkgs.rust-bin.fromRustupToolchain (
    rustToolchainConfig
    // {
      targets = [ target ];
    }
  );
  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
  version = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).package.version;
in
rustPlatform.buildRustPackage {
  pname = "ember-pgo-profile-${arch}";
  inherit version;

  src = import ./ember-source.nix { inherit lib; };
  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    crossCc
    pkgs.buildPackages.binutils
    pkgs.llvmPackages.llvm
  ]
  ++ pkgs.lib.optionals (emulator != "" || arch == "arm64") [ pkgs.qemu-user ];

  buildPhase = ''
    runHook preBuild
    export CARGO_TARGET_${cargoTargetEnv}_LINKER="${linker}"
    export CC_${builtins.replaceStrings [ "-" ] [ "_" ] target}="${linker}"
    export AR_${builtins.replaceStrings [ "-" ] [ "_" ] target}="${archiver}"
    export RUSTFLAGS="-C target-cpu=${targetCpu} -C target-feature=+crt-static -Cprofile-generate=$PWD/profraw"
    mkdir -p profraw
    cargo build --frozen --release --bin ember --target ${target}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/training"
    ${
      if arch == "arm64" then
        ''
          for cpu in cortex-a53 max; do
            printf 'bench depth 12\nbench depth 14\nquit\n' \
              | LLVM_PROFILE_FILE="$PWD/profraw/$cpu-%p.profraw" \
                ${pkgs.qemu-user}/bin/qemu-aarch64 -cpu "$cpu" \
                target/${target}/release/ember >"$out/training/$cpu.log" 2>&1
          done
        ''
      else
        ''
          printf 'bench depth 12\nbench depth 14\nquit\n' \
            | ${emulator}target/${target}/release/ember >"$out/training/bench.log" 2>&1
        ''
    }
    ${pkgs.llvmPackages.llvm}/bin/llvm-profdata merge \
      -o merged.profdata profraw/*.profraw
    test -s merged.profdata || {
      echo "PGO profile merge produced no data" >&2
      exit 1
    }
    ${lib.optionalString (arch == "arm64") ''
      # --covered lists names with nonzero counters (for both IR and frontend
      # profiles). It ignores --function, so filter its complete name list here.
      ${pkgs.llvmPackages.llvm}/bin/llvm-profdata show --covered merged.profdata \
        >"$out/training/covered-functions.txt"
      for kernel in fc0_forward_aarch64_neon fc0_forward_aarch64_dotprod \
        dot_product_aarch64_neon dot_product_aarch64_dotprod; do
        ${pkgs.llvmPackages.llvm}/bin/llvm-profdata show --counts --function="$kernel" \
          merged.profdata >"$out/training/$kernel.counts"
        grep -Fq "$kernel" "$out/training/covered-functions.txt" || {
          echo "PGO workload did not execute $kernel" >&2
          cat "$out/training/$kernel.counts" >&2
          exit 1
        }
      done
    ''}
    mkdir -p "$out"
    cp merged.profdata "$out/merged.profdata"
    runHook postInstall
  '';

  doCheck = false;
  dontFixup = true;

  passthru = {
    inherit arch target;
    workload = "bench depth 12; bench depth 14";
    cpuModels =
      if arch == "arm64" then
        [
          "cortex-a53"
          "max"
        ]
      else
        [ ];
  };

  meta = {
    description = "LLVM PGO profile for Ember ${arch} release builds";
    mainProgram = "ember";
  };
}
