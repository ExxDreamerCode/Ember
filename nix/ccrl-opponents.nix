{ pkgs }:

let
  inherit (pkgs) lib stdenv;

  onlyX86_64 = attrs: lib.optionalAttrs stdenv.hostPlatform.isx86_64 attrs;

  githubSrc = { owner, repo, rev, hash }:
    pkgs.fetchFromGitHub {
      inherit owner repo rev hash;
    };

  seawall-20250322 = stdenv.mkDerivation {
    pname = "ccrl-seawall";
    version = "20250322";

    src = githubSrc {
      owner = "petur";
      repo = "seawall";
      rev = "r20250322";
      hash = "sha256-4TOr3NEfb4AxoGIr6JwAjwGJFXghoW4HAzxUb2Uan2c=";
    };

    nativeBuildInputs = [ pkgs.makeWrapper ];

    buildPhase = ''
      runHook preBuild
      $CXX -Wall -Wextra -std=c++17 -O3 -ffast-math -ftree-vectorize \
        -march=x86-64 -mtune=generic -flto -fno-rtti -fno-exceptions \
        -DSEAWALL_VERSION=20250322 seawall.cc -o seawall
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      install -Dm755 seawall "$out/bin/ccrl-seawall-20250322"
      runHook postInstall
    '';

    meta = {
      description = "Seawall 20250322 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/petur/seawall";
      license = lib.licenses.gpl3Only;
      platforms = lib.platforms.linux;
    };
  };

  olithink-5-11-9 = stdenv.mkDerivation {
    pname = "ccrl-olithink";
    version = "5.11.9-uci";

    src = githubSrc {
      owner = "olithink";
      repo = "OliThink";
      rev = "91577a85bfeb17205bafe7b75114ece6f5b20bed";
      hash = "sha256-Jpw2ljOPvLLoOxzZh6+utMY6radgj96nZlQgYXoJnwk=";
    };

    nativeBuildInputs = [ pkgs.clang ];

    buildPhase = ''
      runHook preBuild
      clang -O3 -Wall -Wextra -mavx2 src/olithink.c -o olithink
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      install -Dm755 olithink "$out/bin/ccrl-olithink-5.11.9"
      runHook postInstall
    '';

    meta = {
      description = "OliThink 5.11.9 UCI chess engine, built from the exact upstream UCI commit";
      homepage = "https://github.com/olithink/OliThink";
      license = lib.licenses.gpl3Plus;
      platforms = [ "x86_64-linux" ];
    };
  };

  byte-knight-4-0-0 = pkgs.rustPlatform.buildRustPackage {
    pname = "ccrl-byte-knight";
    version = "4.0.0";

    src = githubSrc {
      owner = "ptsouchlos";
      repo = "byte-knight";
      rev = "v4.0.0";
      hash = "sha256-rF0BiVDpVpnNfvyZICW6VA51boK8MpONcrJw2ok3do8=";
    };

    cargoHash = "sha256-luz5QRYdXG0Eoh7TPk+0Pezum1e07n1rq057peR00DA=";
    cargoBuildFlags = [ "-p" "byte-knight" "--bin" "byte-knight" ];
    cargoTestFlags = [ "-p" "byte-knight" "--bin" "byte-knight" ];

    postInstall = ''
      mv "$out/bin/byte-knight" "$out/bin/ccrl-byte-knight-4.0.0"
    '';

    meta = {
      description = "byte-knight 4.0.0 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/ptsouchlos/byte-knight";
      license = lib.licenses.gpl3Only;
      platforms = lib.platforms.linux;
    };
  };

  rengar-2-1-1 = stdenv.mkDerivation {
    pname = "ccrl-rengar";
    version = "2.1.1";

    src = githubSrc {
      owner = "teswayze";
      repo = "rengar";
      rev = "v2.1.1";
      hash = "sha256-XzYGa0CIk5xjl2yc4Iyhx70Ni36DwoQxOVorDVc2Png=";
    };

    nativeBuildInputs = [ pkgs.eigen pkgs.gnumake ];

    preBuild = ''
      ln -s ${pkgs.eigen}/include/eigen3/Eigen src/external/Eigen
      touch .EIGEN_INSTALLED
    '';

    makeFlags = [
      "release"
      "arch=x86-64"
      "version=2.1.1"
    ];

    installPhase = ''
      runHook preInstall
      install -Dm755 uci "$out/bin/ccrl-rengar-2.1.1"
      runHook postInstall
    '';

    meta = {
      description = "Rengar 2.1.1 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/teswayze/rengar";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  };

  pawnstar-0-13-593 = stdenv.mkDerivation {
    pname = "ccrl-pawnstar";
    version = "0.13.593";

    src = githubSrc {
      owner = "jonny-reckless";
      repo = "pawnstar";
      rev = "v0.13.593";
      hash = "sha256-9/7AByQKxdWrPxLlBNUZlc1rt0mvByIbTQ8jBeWwns8=";
    };

    nativeBuildInputs = [ pkgs.clang pkgs.gnumake ];

    makeFlags = [
      "RELEASE=1"
      "BUILD_NUMBER=593"
    ];

    installPhase = ''
      runHook preInstall
      install -Dm755 build/pawnstar "$out/bin/ccrl-pawnstar-0.13.593"
      runHook postInstall
    '';

    meta = {
      description = "Pawnstar 0.13.593 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/jonny-reckless/pawnstar";
      license = lib.licenses.gpl3Only;
      platforms = [ "x86_64-linux" ];
    };
  };

  eidolon-1-0-0 = pkgs.rustPlatform.buildRustPackage {
    pname = "ccrl-eidolon";
    version = "1.0.0";

    src = githubSrc {
      owner = "Daniel729";
      repo = "Eidolon";
      rev = "v1.0.0";
      hash = "sha256-3TUJj+CV8zmoaLC8p8M0qOFT37UA9z9HIqPOueBvnzA=";
    };

    cargoHash = "sha256-RTkwK1JUkj+4+z0njXQgPrNaZN5cgToFp+laMFu5LsE=";
    buildAndTestSubdir = "eidolon-bin";
    cargoBuildFlags = [ "--bin" "eidolon" ];
    cargoTestFlags = [ "--bin" "eidolon" ];
    RUSTC_BOOTSTRAP = "1";

    postPatch = ''
      substituteInPlace eidolon-lib/src/lib.rs \
        --replace-fail '#![feature(avx512_target_feature)]' ""
    '';

    postInstall = ''
      mv "$out/bin/eidolon" "$out/bin/ccrl-eidolon-1.0.0"
    '';

    meta = {
      description = "Eidolon 1.0.0 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/Daniel729/Eidolon";
      license = lib.licenses.gpl3Only;
      platforms = [ "x86_64-linux" ];
    };
  };

  puffin-5-0 = pkgs.buildDotnetModule {
    pname = "ccrl-puffin";
    version = "5.0";

    src = githubSrc {
      owner = "kurt1288";
      repo = "Puffin";
      rev = "5.0";
      hash = "sha256-y4XSQfqDjg6rmIlzpU9S24ne2HIOdiobLL5MrC4NmuQ=";
    };

    projectFile = "Puffin.csproj";
    nugetDeps = ./puffin-5.0-deps.json;
    dotnet-sdk = pkgs.dotnet-sdk_8;
    dotnet-runtime = pkgs.dotnet-runtime_8;
    runtimeId = "linux-x64";
    executables = [ "Puffin-5.0" ];

    dotnetRestoreFlags = [ "-p:Platform=x64" ];
    dotnetBuildFlags = [ "-p:Platform=x64" ];
    dotnetInstallFlags = [ "-p:Platform=x64" ];

    postFixup = ''
      mv "$out/bin/Puffin-5.0" "$out/bin/ccrl-puffin-5.0"
    '';

    meta = {
      description = "Puffin 5.0 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/kurt1288/Puffin";
      license = lib.licenses.gpl3Only;
      platforms = [ "x86_64-linux" ];
    };
  };

  revolver-2-0 = stdenv.mkDerivation {
    pname = "ccrl-revolver";
    version = "2.0";

    src = githubSrc {
      owner = "GoldenRare";
      repo = "Revolver";
      rev = "Revolver_2.0";
      hash = "sha256-xkA+o60z3iD3UsjM5EAyxvh/I0T/rgrObGP8TMpCdpU=";
    };

    # The upstream Revolver_2.0 tag still reports "id name Revolver 1.0".
    # Keep the source exact for replay rather than patching the UCI banner.
    buildPhase = ''
      runHook preBuild
      make CC="$CC" \
        CFLAGS="-std=c2x -D_POSIX_C_SOURCE=200809L -pedantic -Wall -Wextra -Wshadow -Wcast-qual -O3 -march=x86-64 -mtune=generic -mavx2 -mbmi -mbmi2 -mpopcnt -flto" \
        LDFLAGS="-std=c2x -D_POSIX_C_SOURCE=200809L -pedantic -Wall -Wextra -Wshadow -Wcast-qual -O3 -march=x86-64 -mtune=generic -mavx2 -mbmi -mbmi2 -mpopcnt -flto"
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      install -Dm755 Revolver "$out/bin/ccrl-revolver-2.0"
      runHook postInstall
    '';

    meta = {
      description = "Revolver 2.0 UCI chess engine, built from the upstream source tag";
      homepage = "https://github.com/GoldenRare/Revolver";
      license = lib.licenses.gpl3Only;
      platforms = [ "x86_64-linux" ];
    };
  };

  knightx-4-92 = stdenv.mkDerivation {
    pname = "ccrl-knightx";
    version = "4.92";

    src = pkgs.fetchurl {
      url = "http://technochess.free.fr/Archive/Knightx492.zip";
      hash = "sha256-TDSWg1wNVG0GQiTXZYXVDnX2hzxlZC4/q6Xl7LnE0U8=";
    };

    nativeBuildInputs = [ pkgs.makeWrapper pkgs.unzip ];

    unpackPhase = ''
      runHook preUnpack
      mkdir source
      cd source
      unzip -q "$src"
      runHook postUnpack
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p "$out/share/ccrl-knightx-4.92"
      cp -R . "$out/share/ccrl-knightx-4.92/"
      chmod +x "$out/share/ccrl-knightx-4.92/knightx492_linux"
      makeWrapper "$out/share/ccrl-knightx-4.92/knightx492_linux" "$out/bin/ccrl-knightx-4.92" \
        --chdir "$out/share/ccrl-knightx-4.92"
      runHook postInstall
    '';

    meta = {
      description = "KnightX 4.92 UCI chess engine, packaged from the upstream Linux/Windows archive";
      homepage = "http://technochess.free.fr/";
      platforms = [ "x86_64-linux" ];
    };
  };

in
onlyX86_64 {
  ccrl-seawall-20250322 = seawall-20250322;
  ccrl-olithink-5-11-9 = olithink-5-11-9;
  ccrl-byte-knight-4-0-0 = byte-knight-4-0-0;
  ccrl-rengar-2-1-1 = rengar-2-1-1;
  ccrl-pawnstar-0-13-593 = pawnstar-0-13-593;
  ccrl-eidolon-1-0-0 = eidolon-1-0-0;
  ccrl-puffin-5-0 = puffin-5-0;
  ccrl-revolver-2-0 = revolver-2-0;
  ccrl-knightx-4-92 = knightx-4-92;

  ccrl-opponents = pkgs.symlinkJoin {
    name = "ccrl-opponents";
    paths = [
      seawall-20250322
      olithink-5-11-9
      byte-knight-4-0-0
      rengar-2-1-1
      pawnstar-0-13-593
      eidolon-1-0-0
      puffin-5-0
      revolver-2-0
      knightx-4-92
    ];
  };
}
