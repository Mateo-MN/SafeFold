import os
import subprocess
from pathlib import Path
from Bio.PDB import PDBParser

def ORF_to_pdb(sequence, outdir=Path(__file__).parent / "ESM_pred"):
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

    pdb_path = None
    for file in os.listdir(outdir):
        if file.endswith(".pdb"):
            pdb_path = os.path.join(outdir, file)
            break

    if pdb_path:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

    for file in os.listdir(outdir):
        file_path = os.path.join(outdir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return structure[0]


if __name__ == "__main__":
    seq = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
    print(ORF_to_pdb(seq))