import subprocess
import os
from pathlib import Path


def alphafold_pred(sequence, outdir="results"):
    Path(outdir).mkdir(exist_ok=True)

    cmd = [
        "amina",
        "run",
        "esmfold",
        "--sequence",
        sequence,
        "-o",
        outdir
    ]

    subprocess.run(cmd, check=True)

    # find generated pdb
    for file in os.listdir(outdir):
        if file.endswith(".pdb"):
            return os.path.join(outdir, file)

    raise RuntimeError("No PDB produced")


if __name__ == "__main__":
    seq = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
    
    
    pdb_path = alphafold_pred(seq)

    # print file contents
    with open(pdb_path, "r") as f:
        print(f.read())

    os.remove(pdb_path)