"""
Caffe Model Compatibility Checker and Upgrader

This script checks whether the specified Caffe .prototxt and .caffemodel are compatible
and upgrades them if necessary, saving updated versions with "_updated" postfixes.

Usage:
    python caffe_model_checker.py --prototxt=<path_to_prototxt> --caffemodel=<path_to_caffemodel>
"""

import os
import sys
import argparse
import subprocess
import caffe


def upgrade_file(file_path: str, tool: str, output_postfix: str) -> str:
    """
    Upgrade a Caffe file using the corresponding C++ tool.

    Parameters
    ----------
    file_path : str
        Path to the file to be upgraded (.prototxt or .caffemodel).
    tool : str
        The name of the Caffe upgrade tool (e.g., "upgrade_net_proto_text").
    output_postfix : str
        Postfix to add to the upgraded file.

    Returns
    -------
    str
        Path to the upgraded file.
    """
    output_path = os.path.splitext(file_path)[0] + output_postfix
    try:
        subprocess.run([tool, file_path, output_path], check=True)
        print(f"{file_path} successfully upgraded and saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade {file_path} using {tool}: {e}")
        return file_path  # Return original path if upgrade fails


def upgrade_and_check(prototxt_path: str, caffemodel_path: str) -> None:
    """
    Load the Caffe network, check for compatibility, and upgrade files if needed.

    Parameters
    ----------
    prototxt_path : str
        Path to the .prototxt file defining the network architecture.
    caffemodel_path : str
        Path to the .caffemodel file containing the trained weights.

    Returns
    -------
    None
    """
    # Load the network
    try:
        net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    except Exception as e:
        print(f"Error loading network: {e}")
        return

    # Check if layers match
    prototxt_layers = net.params.keys()
    caffemodel_layers = net.params.keys()

    print("Layers in prototxt:", prototxt_layers)
    print("Layers in caffemodel:", caffemodel_layers)

    if prototxt_layers == caffemodel_layers:
        print("The .prototxt and .caffemodel match.")
    else:
        print("Mismatch between .prototxt and .caffemodel.")

    # Upgrade files if needed
    updated_prototxt = upgrade_file(
        prototxt_path, "upgrade_net_proto_text", "_updated.prototxt"
    )
    updated_caffemodel = upgrade_file(
        caffemodel_path, "upgrade_net_proto_binary", "_updated.caffemodel"
    )

    print(f"Upgraded files saved to: {updated_prototxt}, {updated_caffemodel}")
    print("Upgrade process completed.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check and upgrade Caffe .prototxt and .caffemodel files for compatibility."
    )
    parser.add_argument(
        "--prototxt",
        type=str,
        required=True,
        help="Path to the .prototxt file defining the network architecture.",
    )
    parser.add_argument(
        "--caffemodel",
        type=str,
        required=True,
        help="Path to the .caffemodel file containing the trained weights.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to handle command-line arguments and process the files.

    Returns
    -------
    None
    """
    args = parse_arguments()

    if not os.path.exists(args.prototxt):
        print(f"Error: Prototxt file '{args.prototxt}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.caffemodel):
        print(f"Error: Caffemodel file '{args.caffemodel}' does not exist.")
        sys.exit(1)

    upgrade_and_check(args.prototxt, args.caffemodel)


if __name__ == "__main__":
    main()
