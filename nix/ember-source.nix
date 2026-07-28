{ lib }:

lib.cleanSourceWith {
  src = ../.;
  filter =
    path: type:
    let
      root = toString ../.;
      relative = lib.removePrefix "${root}/" (toString path);
    in
    toString path == root
    || relative == "Cargo.toml"
    || relative == "Cargo.lock"
    || relative == "src"
    || lib.hasPrefix "src/" relative;
}
