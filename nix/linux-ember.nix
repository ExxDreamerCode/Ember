{
  pkgs,
  lib,
  arch,
}:

let
  target =
    {
      amd64 = "x86_64-unknown-linux-musl";
      arm64 = "aarch64-unknown-linux-musl";
    }
    .${arch} or (throw "unsupported Linux Ember architecture: ${arch}");
  targetCpu =
    {
      amd64 = "x86-64-v3";
      arm64 = "generic";
    }
    .${arch};
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
  pname = "ember-linux-${arch}";
  inherit version;

  src = import ./ember-source.nix { inherit lib; };
  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    crossCc
    pkgs.buildPackages.binutils
    pkgs.buildPackages.file
  ];

  buildPhase = ''
    runHook preBuild
    export CARGO_TARGET_${cargoTargetEnv}_LINKER="${linker}"
    export CC_${builtins.replaceStrings [ "-" ] [ "_" ] target}="${linker}"
    export AR_${builtins.replaceStrings [ "-" ] [ "_" ] target}="${archiver}"
    export RUSTFLAGS="-C target-cpu=${targetCpu} -C target-feature=+crt-static"
    cargo build --frozen --release --bin ember --target ${target}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"
    cp "target/${target}/release/ember" "$out/bin/ember"

    file "$out/bin/ember" | tee "$out/FILE.txt"
    if readelf -l "$out/bin/ember" | grep -q 'INTERP'; then
      echo "Linux release binary has a dynamic interpreter" >&2
      exit 1
    fi
    if readelf -d "$out/bin/ember" 2>/dev/null | grep -q 'NEEDED'; then
      echo "Linux release binary has dynamic library dependencies" >&2
      exit 1
    fi
    runHook postInstall
  '';

  doCheck = false;
  dontFixup = true;

  passthru = {
    inherit arch target targetCpu;
    allocator = if arch == "amd64" then "mimalloc" else "system";
    linkage = "static-musl";
  };

  meta = {
    description = "Static Ember chess engine for Linux ${arch}";
    mainProgram = "ember";
  };
}
