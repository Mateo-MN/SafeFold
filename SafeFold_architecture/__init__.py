import os
from dotenv import load_dotenv
from SafeFold_architecture.Layers.ORF_detector import find_orfs

load_dotenv()
api_key = os.getenv("AMINA_API_KEY")

def analyseDNA(DNA):
    ORFs = find_orfs(DNA)
    for ORF in ORFs:
        print(ORF)
    