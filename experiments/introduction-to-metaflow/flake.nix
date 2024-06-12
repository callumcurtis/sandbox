{
  description = "Introduction to Metaflow";

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
          packages = with pkgs; [
            (writeShellScriptBin "python" ''
              export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
              exec ${python313}/bin/python "$@"
            '')
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
