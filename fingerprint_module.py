from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def generate_mol_object(dataset_df):
    array_of_smiles = np.concatenate(dataset_df.values)
#    num_molecules = np.size(array_of_smiles)
    mol_objects = list(filter(lambda x: x[1] is not None,((idx, Chem.MolFromSmiles(smile_code)) for idx, smile_code in enumerate(array_of_smiles)))) #skips over the ones that do not convert

    # Now mol objects contains tuples of the index of the mol object and the object. This is done in case that some of the conversions of smiles to mol fail, to keep track of the remaining
    # smiles that converted correctly and select the appropriate y values.
    return mol_objects

def generate_morgan_fp(dataset_df, radius, bits):
    mols = generate_mol_object(dataset_df)
    morgan_fps = [ 
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
        for idx, mol in mols
        ]
    
    #Convert in numpy array
    morgan_fps_np = np.array([np.array(fp) for fp in morgan_fps])
    return morgan_fps_np

def filter_labels(y_labels, dataset_df): #selects the labes of only those smiles that converted
    # Perform the conversion to get indexes
    mol_tuples = generate_mol_object(dataset_df)

    #In numpy (for everyone's sanity) extract the features corresponding to well converted smiles
    y_labels_numpy = y_labels.to_numpy()
    selected_y = [y_labels_numpy[i] for i, _ in mol_tuples]
    return selected_y
