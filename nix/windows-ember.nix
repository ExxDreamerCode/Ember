{
  pkgs,
  lib,
  arch ? "amd64",
}:

let
  target =
    {
      amd64 = "x86_64-pc-windows-msvc";
      arm64 = "aarch64-pc-windows-msvc";
    }
    .${arch} or (throw "unsupported Windows Ember architecture: ${arch}");
  xwinConfig = {
    arch =
      {
        amd64 = "x86_64";
        arm64 = "aarch64";
      }
      .${arch};
    variant = "desktop";
    version = "17";
    sdkVersion = "10.0.26100";
    # xwin expects the manifest toolset selector, not the 14.44.35220
    # build number reported by the installed CRT headers.
    crtVersion = "14.44.17.14";
    cacheHash =
      {
        amd64 = "sha256-wHNCHGHGJcKv+oN/sDceNBitjQdwRcEWOZlJnu/CzSE=";
        arm64 = "sha256-ejLSpOVpszYXdfovAfhVRyyn/cVgXNwCiKQao+smags=";
      }
      .${arch};
  };
  defaultTargetCpu =
    {
      amd64 = "x86-64-v3";
      arm64 = "generic";
    }
    .${arch};
  releaseAppName = if arch == "amd64" then "windows-release" else "windows-release-arm64";
  rustToolchainConfig = (builtins.fromTOML (builtins.readFile ../rust-toolchain.toml)).toolchain;
  windowsRustToolchain = pkgs.rust-bin.fromRustupToolchain (
    rustToolchainConfig
    // {
      targets = [ target ];
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
    --target ${target}
  '';
  rustFlags = "-C target-cpu=${defaultTargetCpu} -C target-feature=+crt-static -C link-arg=/STACK:16777216";

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
  version = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).package.version;

  emberWindows = rustPlatform.buildRustPackage {
    pname = "ember-windows-${arch}";
    inherit version;

    src = import ./ember-source.nix { inherit lib; };

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
      export PATH="${pkgs.llvmPackages.clang-unwrapped}/bin:$PATH"
      mkdir -p "$HOME" "$XWIN_CACHE_DIR"
      cp -R "${xwinSdk}/." "$XWIN_CACHE_DIR/"
      chmod -R u+w "$XWIN_CACHE_DIR"
      export RUSTFLAGS="${rustFlags}"
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
      cp "target/${target}/release/ember.exe" "$out/bin/ember.exe"
      runHook postInstall
    '';

    doCheck = false;
    dontFixup = true;

    passthru = {
      inherit
        arch
        target
        windowsRustToolchain
        xwinConfig
        xwinSdk
        ;
      targetCpu = defaultTargetCpu;
      allocator = if arch == "amd64" then "mimalloc" else "system";
      linkage = "static-msvc-crt";
    };
  };

  # Compatibility frontend for the original `nix run .#windows-release`
  # workflow. Its xwin pins and Cargo arguments come from the same definitions
  # as the pure `windows-ember` package above.
  releaseApp = pkgs.writeShellApplication {
    name = releaseAppName;
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
      export PATH="${pkgs.llvmPackages.clang-unwrapped}/bin:$PATH"

      target_cpu="''${EMBER_WINDOWS_TARGET_CPU:-${defaultTargetCpu}}"
      export RUSTFLAGS="-C target-cpu=$target_cpu -C target-feature=+crt-static -C link-arg=/STACK:16777216 ''${EMBER_WINDOWS_RUSTFLAGS:-}"
      cargo_xwin_args=(
        ${cargoBuildArrayItems}
      )
      exec cargo-xwin build "''${cargo_xwin_args[@]}" "$@"
    '';
  };
in
{
  package = emberWindows;
  inherit
    releaseApp
    windowsRustToolchain
    xwinConfig
    xwinSdk
    ;
}
