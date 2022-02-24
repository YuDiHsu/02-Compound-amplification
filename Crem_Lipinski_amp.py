import itertools
import os

import pandas as pd
from crem.crem import grow_mol, link_mols, mutate_mol
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import Lipinski


def _mutate_replace(m, db_path):
    r_list = []
    for i in range(1, 4):
        # r = list(grow_mol(m, db_name=db_path, min_atoms=1, radius=i, max_atoms=5))
        r = list(mutate_mol(Chem.AddHs(m), db_name=db_path, radius=i, max_rel_size=2, max_inc=2))
        r_list.append(r)

    return r_list


def _grow_replace(m, db_path):
    r_list = []
    for i in range(1, 4):
        r = list(grow_mol(m, db_name=db_path, min_atoms=1, radius=i, max_atoms=2))
        r_list.append(r)

    return r_list


def _link_fragments(m, m_2, db_path):
    return list(link_mols(m, m2, db_name=db_path, radius=2, min_atoms=1, max_atoms=2, max_replacements=3))


class SmilesError(Exception):
    pass


def _log_partition_coefficient(smiles):
    '''
    Returns the octanol-water partition coefficient given a molecule SMILES
    string
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        raise SmilesError(f'{smiles} returns a None molecule')
    return Crippen.MolLogP(mol)


def _lipinski_trial(smiles, **rules_dict):
    # #  Returns which of Lipinski's rules a molecule has failed, or an empty list
    '''
    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 12
    Molecular weight < 600 daltons
    logP < 6
    '''
    rules_dict.setdefault('hbd', 5)
    rules_dict.setdefault('hba', 12)
    rules_dict.setdefault('mw', 600)
    rules_dict.setdefault('logp', 6)
    rules_dict.setdefault('rotb', 14)
    rules_dict.setdefault('tpsa', 208)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception(f'{smiles} is not a valid SMILES string')

    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Crippen.MolLogP(mol)
    num_rotb = Lipinski.NumRotatableBonds(mol)
    mol_tpsa = Descriptors.TPSA(mol)

    info_mol = dict(Smiles_code=smiles, HBD_value=num_hdonors, HBD_validation='Failed',
                    HBA_value=num_hacceptors, HBA_validation='Failed', MW_value=mol_weight, MW_validation='Failed',
                    LogP_value=mol_logp, LogP_validation='Failed', RotB_value=num_rotb, RotB_validation='Failed',
                    TPSA_value=mol_tpsa, TPSA_validation='Failed')

    passed = []
    failed = []
    necessary_counts = 0
    if num_hdonors > rules_dict['hbd']:
        failed.append(f"Over {rules_dict['hbd']} H-bond donors, found {num_hdonors}")
    else:
        info_mol['HBD_validation'] = 'Passed'
        passed.append(f"Found {num_hdonors} H-bond donors")
        necessary_counts += 1

    if num_hacceptors > rules_dict['hba']:
        failed.append(f"Over {rules_dict['hba']} H-bond acceptors, found {num_hacceptors}")
    else:
        info_mol['HBA_validation'] = 'Passed'
        passed.append(f"Found {num_hacceptors} H-bond acceptors")

    if mol_weight > rules_dict['mw']:
        failed.append(f"Molecular weight over {rules_dict['mw']}, calculated {mol_weight}")
    else:
        info_mol['MW_validation'] = 'Passed'
        passed.append(f"Molecular weight: {mol_weight}")
        necessary_counts += 1

    if mol_logp > rules_dict['logp']:
        failed.append(f"Log partition coefficient over {rules_dict['logp']}, calculated {mol_logp}")
    else:
        info_mol['LogP_validation'] = 'Passed'
        passed.append(f"Log partition coefficient: {mol_logp}")
        necessary_counts += 1

    if num_rotb > rules_dict['rotb']:
        failed.append(f"Rotatable Bonds of molecule over {rules_dict['num_rotb']}, calculated {num_rotb}")
    else:
        info_mol['RotB_validation'] = 'Passed'
        passed.append(f"Rotatable Bonds of molecule: {num_rotb}")

    if mol_tpsa > rules_dict['tpsa']:
        failed.append(f"Topological polar surface area over {rules_dict['tpsa']}, calculated {mol_tpsa}")
    else:
        info_mol['TPSA_validation'] = 'Passed'
        passed.append(f"Topological polar surface area: {mol_tpsa}")

    if necessary_counts == 3:
        return True, info_mol


def _lipinski_pass(smiles):
    # # Wraps around lipinski trial, but returns a simple pass/fail True/False
    passed, failed = _lipinski_trial(smiles)
    if failed:
        return False
    else:
        return True


if __name__ == '__main__':
    smi = 'CC(C)(O)C1CC2=C3C(C(C(C(C=C(O)C=C4)=C4CC5)=C5O3)=O)=C(O)C=C2O1'  # TCI_DZ_10S
    # smi = 'COC1=CC=CC2=C1C(=CN2)CC#N'  # TCI_04S
    m = Chem.MolFromSmiles(smi)  # methoxytoluene
    path = os.path.join('.', 'replacements02_sc2.db')
    m2 = Chem.MolFromSmiles('NCC(=O)O')  # glycine

    mols_1 = _mutate_replace(m, path)
    mols_2 = _grow_replace(m, path)
    # mols_3 = _link_fragments(m, m2, path)

    n = 0
    passed_mol_list = []
    total_mol_set = set(itertools.chain(*(mols_1 + mols_2)))

    for m in total_mol_set:
        lipinski_status, passed_mol = _lipinski_trial(m)
        if lipinski_status:
            n += 1
            passed_mol_list.append(passed_mol)
            fig = Chem.MolFromSmiles(m)
            draw = Draw.MolToImage(fig)
            draw.save(os.path.join('.', 'exported_data', f'{n}.jpg'))
    print(n)
    df = pd.DataFrame(passed_mol_list, index=None)
