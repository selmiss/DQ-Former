from rdkit import Chem
from rdkit.Chem import Draw

# Define two groups as small molecules for visualization
mol1 = Chem.MolFromSmiles("CCN")  # acetic acid (CH3-COOH), represents –CH2–C(=O)OH arm
mol2 = Chem.MolFromSmiles("CN")   # formic acid fragment, represents –C(=O)OH

# Increase image resolution and line width for clarity
img = Draw.MolsToImage(
    [mol1, mol2],
    subImgSize=(500, 500),
    fontSize=40,
    legends=["–CH2–C(=O)OH (乙酸基)", "–C(=O)OH (羧基)"]
)

# Save higher resolution figure
img.save("trainer/entropy_model/fig/groups_comparison_highres.png")
