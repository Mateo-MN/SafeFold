import os
import dgl
import esm
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
import scipy.sparse as sp
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# EDIT THESE
PID = "A0S864"
ONTOLOGY = "mf"  # "mf", "bp", "cc"
PDB_PATH = f"./data/PDB/{PID}.pdb"

# Constants from DPFunc_model checkpoints
INTERPRO_DIM = 22369

def save_pkl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def extract_sequence_and_ca_coords(pdb_file, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    if chain_id is None:
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in PDB")
        chain = chains[0]
    else:
        chain = model[chain_id]

    seq = []
    ca_coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        try:
            aa = seq1(residue.resname)
        except Exception:
            continue
        seq.append(aa)
        ca_coords.append(residue["CA"].coord.astype(float))

    if not seq or not ca_coords:
        raise ValueError("Failed to extract sequence/CA coords (check chain selection and PDB completeness).")

    return "".join(seq), np.vstack(ca_coords)

def embed_esm2_t33_650M(seq: str) -> np.ndarray:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("protein", seq)])
    with torch.no_grad():
        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    rep = out["representations"][model.num_layers][0, 1:1+len(seq)]  # [L, 1280]
    return rep.cpu().numpy().astype(np.float32)

def build_graph_from_points(points: np.ndarray, threshold: float = 12.0):
    L = points.shape[0]
    u, v, dis = [], [], []
    for i in range(L):
        pi = points[i]
        for j in range(L):
            if i == j:
                continue
            d = float(np.linalg.norm(pi - points[j]))
            if d <= threshold:
                u.append(i); v.append(j); dis.append(d)

    g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=L)
    g.edata["dis"] = torch.tensor(dis, dtype=torch.float32)
    return g

def get_GO_terms(PDB_PATH: str):
    save_pkl(f"./processed_file/{ONTOLOGY}_test_used_pid_list.pkl", [PID])
    
    seq, coords = extract_sequence_and_ca_coords(PDB_PATH, chain_id=None)
    save_pkl("./processed_file/pdb_points.pkl", {PID: coords})
    save_pkl("./processed_file/pdb_seqs.pkl", {PID: seq})
    
    emb = embed_esm2_t33_650M(seq)
    assert emb.shape[0] == len(seq)
    assert emb.shape[1] == 1280, emb.shape
    save_pkl("./processed_file/esm_emds/esm_part_0.pkl", {PID: emb})
    
    g = build_graph_from_points(coords, threshold=12.0)
    g.ndata["x"] = torch.from_numpy(emb)
    out_graph_path = f"./processed_file/graph_features/{ONTOLOGY}_test_whole_pdb_part0.pkl"
    save_pkl(out_graph_path, [g])

    dummy_go = "GO:0003674"  # molecular_function
    go_path = f"./processed_file/{ONTOLOGY}_test_pid_go.txt"
    with open(go_path, "w") as f:
        f.write(f"{PID}\t{dummy_go}\n")
    
    X = sp.csr_matrix((1, INTERPRO_DIM), dtype=np.float32)
    with open(f"./processed_file/{ONTOLOGY}_test_interpro_file.pkl", "wb") as f:
        pkl.dump(X, f)
    save_pkl(f"./processed_file/interpro/{PID}.pkl", np.zeros(INTERPRO_DIM, dtype=np.float32))
    
    cfg_path = f"./configure/{ONTOLOGY}.yaml"
    yaml_lines = [
        f"name: {ONTOLOGY}\n",
        f"mlb: ./mlb/{ONTOLOGY}_go.mlb\n",
        "results: ./results\n\n",
        "base:\n",
        "  interpro_whole: ./processed_file/interpro/{}.pkl\n\n",
        "test:\n",
        "  name: test\n",
        f"  pid_list_file: ./processed_file/{ONTOLOGY}_test_used_pid_list.pkl\n",
        f"  pid_pdb_file: ./processed_file/graph_features/{ONTOLOGY}_test_whole_pdb_part0.pkl\n",
        f"  pid_go_file: ./processed_file/{ONTOLOGY}_test_pid_go.txt\n",
        f"  interpro_file: ./processed_file/{ONTOLOGY}_test_interpro_file.pkl\n",
    ]
    with open(cfg_path, "w") as f:
        f.writelines(yaml_lines)
    print("Wrote:", cfg_path)

    print(f"Run:\n  python DPFunc_pred.py -d {ONTOLOGY} -n 0 -p DPFunc_model")




Path("./processed_file/graph_features").mkdir(parents=True, exist_ok=True)
Path("./processed_file/esm_emds").mkdir(parents=True, exist_ok=True)
Path("./processed_file/interpro").mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)
Path("./configure").mkdir(parents=True, exist_ok=True)

save_pkl(f"./processed_file/{ONTOLOGY}_test_used_pid_list.pkl", [PID])
print("Wrote:", f"./processed_file/{ONTOLOGY}_test_used_pid_list.pkl")
