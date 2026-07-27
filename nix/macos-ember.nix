{
  pkgs,
  nixpkgs,
  rust-overlay,
  lib,
  arch,
}:

let
  target =
    {
      amd64 = "x86_64-apple-darwin";
      arm64 = "aarch64-apple-darwin";
    }
    .${arch} or (throw "unsupported macOS Ember architecture: ${arch}");
  targetSystem =
    {
      amd64 = "x86_64-darwin";
      arm64 = "aarch64-darwin";
    }
    .${arch};
  targetCpu =
    {
      # Keep the baseline runnable under Rosetta 2. Ember's runtime dispatch
      # still selects its AVX2 search and NNUE backends on capable Intel Macs.
      amd64 = "x86-64";
      arm64 = "apple-m1";
    }
    .${arch};
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
    }
  );
  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
  version = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).package.version;
in
rustPlatform.buildRustPackage {
  pname = "ember-macos-${arch}";
  inherit version;

  src = import ./ember-source.nix { inherit lib; };
  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    targetCc
    pkgs.buildPackages.file
  ];

  buildPhase = ''
    runHook preBuild
    export CARGO_TARGET_${cargoTargetEnv}_LINKER="${linker}"
    export MACOSX_DEPLOYMENT_TARGET=11.0
    export RUSTFLAGS="-C target-cpu=${targetCpu}"
    cargo build --frozen --release --bin ember --target ${target}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"
    cp "target/${target}/release/ember" "$out/bin/ember"

    # The Darwin linker can record Nix's libiconv install name. macOS provides
    # the same stable system library, and release archives must not require a
    # user's Nix store.
    while read -r dependency _; do
      case "$dependency" in
        /nix/store/*-libiconv-*/lib/libiconv.2.dylib)
          install_name_tool -change \
            "$dependency" \
            /usr/lib/libiconv.2.dylib \
            "$out/bin/ember"
          ;;
      esac
    done < <(otool -L "$out/bin/ember" | tail -n +2)

    while read -r dependency _; do
      case "$dependency" in
        /usr/lib/*|/System/Library/*) ;;
        *)
          echo "macOS release binary has a non-system dependency: $dependency" >&2
          exit 1
          ;;
      esac
    done < <(otool -L "$out/bin/ember" | tail -n +2)

    file "$out/bin/ember" | tee "$out/FILE.txt"
    runHook postInstall
  '';

  doCheck = false;
  dontFixup = true;

  passthru = {
    inherit
      arch
      target
      targetCpu
      targetSystem
      ;
    allocator = "system";
    linkage = "system-libSystem";
    minimumMacOS = "11.0";
  };

  meta = {
    description = "Self-contained Ember chess engine for macOS ${arch}";
    mainProgram = "ember";
  };
}
