{
  pkgs,
  lib,
}:

let
  xwinConfig = {
    arch = "x86_64";
    variant = "desktop";
    version = "17";
    sdkVersion = "10.0.26100";
    # xwin expects the manifest toolset selector, not the 14.44.35220
    # build number reported by the installed CRT headers.
    crtVersion = "14.44.17.14";
    cacheHash = "sha256-wHNCHGHGJcKv+oN/sDceNBitjQdwRcEWOZlJnu/CzSE=";
  };
  defaultTargetCpu = "x86-64-v3";
  rustToolchainConfig =
    (builtins.fromTOML (builtins.readFile ../rust-toolchain.toml)).toolchain;
  windowsRustToolchain = pkgs.rust-bin.fromRustupToolchain (
    rustToolchainConfig
    // {
      targets = [ "x86_64-pc-windows-msvc" ];
    }
  );

  xwinEnvironment = ''
    export XWIN_ARCH=${xwinConfig.arch}
    export XWIN_VARIANT=${xwinConfig.variant}
    export XWIN_VERSION=${xwinConfig.version}
    export XWIN_SDK_VERSION=${xwinConfig.sdkVersion}
    export XWIN_CRT_VERSION=${xwinConfig.crtVersion}
  '';
  cargoBuildArrayItems = ''
    --locked
    --release
    --bin ember
    --target x86_64-pc-windows-msvc
  '';

  # Keep cargo-xwin's network access in a fixed-output derivation. Both the
  # exposed Windows package and the portable ZIP use this exact SDK cache.
  xwinSdk = pkgs.stdenvNoCC.mkDerivation {
    pname = "xwin-sdk-cache";
    version = "${xwinConfig.version}-${xwinConfig.sdkVersion}-${xwinConfig.crtVersion}";

    dontUnpack = true;
    nativeBuildInputs = [
      pkgs.cacert
      pkgs.cargo-xwin
    ];

    outputHashAlgo = "sha256";
    outputHashMode = "recursive";
    outputHash = xwinConfig.cacheHash;

    buildPhase = ''
      runHook preBuild
      export HOME="$TMPDIR/home"
      export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      ${xwinEnvironment}
      export XWIN_CACHE_DIR="$out"
      mkdir -p "$HOME" "$out"
      cargo-xwin cache xwin
      runHook postBuild
    '';

    dontInstall = true;
    dontFixup = true;
  };

  rustPlatform = pkgs.makeRustPlatform {
    cargo = windowsRustToolchain;
    rustc = windowsRustToolchain;
  };

  emberWindows = rustPlatform.buildRustPackage {
    pname = "ember-windows";
    version = "1.1.2";

    src = lib.cleanSourceWith {
      src = ../.;
      filter = path: type:
        let
          root = toString ../.;
          relative = lib.removePrefix "${root}/" (toString path);
        in
        toString path == root
        || relative == "Cargo.toml"
        || relative == "Cargo.lock"
        || relative == "src"
        || lib.hasPrefix "src/" relative;
    };

    cargoLock.lockFile = ../Cargo.lock;
    nativeBuildInputs = [
      pkgs.cargo-xwin
      pkgs.clang
      pkgs.lld
      pkgs.llvmPackages.llvm
    ];

    buildPhase = ''
      runHook preBuild
      export HOME="$TMPDIR/home"
      ${xwinEnvironment}
      export XWIN_CACHE_DIR="$TMPDIR/cargo-xwin"
      mkdir -p "$HOME" "$XWIN_CACHE_DIR"
      cp -R "${xwinSdk}/." "$XWIN_CACHE_DIR/"
      chmod -R u+w "$XWIN_CACHE_DIR"
      export RUSTFLAGS="-C target-cpu=${defaultTargetCpu} -C target-feature=+crt-static"
      cargo_xwin_args=(
        --offline
        ${cargoBuildArrayItems}
      )
      cargo-xwin build "''${cargo_xwin_args[@]}"
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p "$out/bin"
      cp target/x86_64-pc-windows-msvc/release/ember.exe "$out/bin/ember.exe"
      runHook postInstall
    '';

    doCheck = false;
    dontFixup = true;

    passthru = {
      inherit windowsRustToolchain xwinConfig xwinSdk;
      targetCpu = defaultTargetCpu;
    };
  };

  # Compatibility frontend for the original `nix run .#windows-release`
  # workflow. Its xwin pins and Cargo arguments come from the same definitions
  # as the pure `windows-ember` package above.
  releaseApp = pkgs.writeShellApplication {
    name = "windows-release";
    runtimeInputs = with pkgs; [
      cargo-xwin
      clang
      lld
      llvmPackages.llvm
      windowsRustToolchain
    ];
    text = ''
      ${xwinEnvironment}
      export XWIN_CACHE_DIR="''${XWIN_CACHE_DIR:-$HOME/.cache/cargo-xwin}"
      export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-target/xwin}"

      target_cpu="''${EMBER_WINDOWS_TARGET_CPU:-${defaultTargetCpu}}"
      export RUSTFLAGS="-C target-cpu=$target_cpu -C target-feature=+crt-static ''${EMBER_WINDOWS_RUSTFLAGS:-}"
      cargo_xwin_args=(
        ${cargoBuildArrayItems}
      )
      exec cargo-xwin build "''${cargo_xwin_args[@]}" "$@"
    '';
  };
in
{
  package = emberWindows;
  inherit releaseApp windowsRustToolchain xwinConfig xwinSdk;
}
