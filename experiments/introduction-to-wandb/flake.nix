{
  description = "Introduction to Wandb";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        venv = "./.venv";
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc
          ]);

          packages = with pkgs; [
            python312
          ];

          shellHook = ''
            if test ! -d ${venv}; then
              python -m venv ${venv}
            fi

            source ${venv}/bin/activate
          '';
        };
      });
}
