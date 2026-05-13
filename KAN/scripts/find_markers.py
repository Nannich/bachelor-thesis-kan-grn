import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_paul15

def main():
    adata = load_paul15()
    
    raw_gene_names = adata.raw.var_names.tolist()
    
    genes = [
        "Gata1", "Klf1",            # Erythroid
        "Mpo", "Elane", "Cebpa",    # Myeloid
        "Cd34", "Gata2"             # Progenitors
    ]
    
    for g in genes:
        idx = raw_gene_names.index(g)
        print(f"{g:<8} -> Index: {idx:<4}")

if __name__ == "__main__":
    main()

"""
Gata1    -> Index: 1252
Klf1     -> Index: 1670
Mpo      -> Index: 1913
Elane    -> Index: 1040
Cebpa    -> Index: 664 
Cd34     -> Index: 619 
Gata2    -> Index: 1253
"""

