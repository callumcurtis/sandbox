{
  description = "Video compressor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flamegraph.url = "git+ssh://git@github.com/callumcurtis/snippets?dir=topic/profiling/flamegraph";
    flamegraph.inputs.nixpkgs.follows = "nixpkgs";
    flamegraph.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { nixpkgs, flake-utils, flamegraph, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            python312Packages.numpy
            python312Packages.scipy
            python312Packages.matplotlib
            python312Packages.pydantic
            ffmpeg_6
            imagemagick
            vlc
            xxd
            flamegraph.packages.${system}.default
          ];
        };
      });
}
