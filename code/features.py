from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import pandas as pd
import numpy as np
from typing import Optional, Callable
import os
import os.path as osp
from typing import Tuple, List, Dict, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from pymatgen.core import Molecule, Structure, Element
from pymatgen.core.bonds import CovalentBond
from pymatgen.io.babel import BabelMolAdaptor


## removing warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


import torch
import torch.nn.functional as F
from collections import defaultdict

AVERAGE_ELECTRONEGATIVITY=1.713


#def node_features(mol:Chem.rdchem.Mol):
#    # 3. QM9 featurization of nodes
#    N = mol.GetNumAtoms()
#    atomic_number = []
#    aromatic = []
#    sp = []
#    sp2 = []
#    sp3 = []
#    for atom in mol.GetAtoms():
#        atomic_number.append(atom.GetAtomicNum())
#        aromatic.append(1 if atom.GetIsAromatic() else 0)
#        hybridization = atom.GetHybridization()
#        sp.append(1 if hybridization == HybridizationType.SP else 0)
#        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
#        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
#    x = torch.tensor([atomic_number, aromatic, sp, sp2, sp3],dtype=torch.float).t().contiguous()
#    z = torch.tensor(atomic_number, dtype=torch.long)
#    
#    return x,z


def from_rdkit_mol_to_pymatgen_mol(mol: Chem.rdchem.Mol) -> Molecule:
    
    try:
        ## expects an rdkit mol with 3D conformer
        mol_file = Chem.MolToMolBlock(mol)
        pymatgen_mol = BabelMolAdaptor.from_string(string_data=mol_file, file_format="mol").pymatgen_mol
    except:
        return None
        
    
    return pymatgen_mol




def pymatgen_node_features(data: Union[Chem.rdchem.Mol,Structure]) -> Tuple[torch.Tensor, torch.Tensor]:
    
    ## transform molecule or structure into list of atomic elements
    if isinstance(data, Chem.rdchem.Mol):
        elem_list = []
        for atom in data.GetAtoms():
            z = atom.GetAtomicNum()
            elem= Element.from_Z(z)
            elem_list.append(elem)
        
    elif isinstance(data, Structure):
        elem_list = data.species
        
    ## defines elemental features    
    features = ["atomic_radius","atomic_mass","average_ionic_radius", "average_cationic_radius", "average_anionic_radius", "max_oxidation_state",
            "min_oxidation_state", "row","group", "is_noble_gas", "is_post_transition_metal", "is_rare_earth_metal", "is_metal", "is_metalloid",
            "is_alkali", "is_alkaline", "is_halogen","is_chalcogen", "is_lanthanoid","is_actinoid", "is_quadrupolar"] 
    
    features_dict = {feature:[] for feature in features}
    


    atomic_number = []
    electronegativity = []
    for elem in elem_list:
        atomic_number.append(elem.Z)
        #X = elem.X
        #if np.isnan(x):
        #    X = AVERAGE_ELECTRONEGATIVITY
        
        #electronegativity.append(X)
        for feature in features:
            features_dict[feature].append(getattr(elem,feature))
    
    
    
    all_features = [feature_list for feature_list in features_dict.values()]
    x= [atomic_number] + all_features
    x = torch.tensor(x,dtype=torch.float).t().contiguous()
    z = torch.tensor(atomic_number, dtype=torch.long)
    
    return x,z



def edge_features(data: Union[Chem.rdchem.Mol,Structure]) -> Tuple[torch.Tensor, torch.Tensor]:
    
    #4. Create the complete graph (no self-loops) with covalent bond types as edge attributes
    
    bonds_type = {"no-bond":0,"single": 1, "no-data": 2}

    if isinstance(data, Chem.rdchem.Mol):
        mol=data
        # getting all covalent bond types
        bonds_dict = {(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()):bonds_type["single"] for bond in mol.GetBonds()}
        # returns 0 for all pairs of atoms with no covalent bond 
        bonds_dict = defaultdict(int, bonds_dict)
        N = mol.GetNumAtoms()
        
        
    elif isinstance(data, Structure):
        struct=data
        N = len(struct.species)

    # making the complete graph
    first_node_index = []
    second_node_index = []
    edge_type=[]
        
    ## building complete graph edge indexes
    is_molecule = isinstance(data, Chem.rdchem.Mol)
    is_structure = isinstance(data, Structure)
    for i in range(N):
        for j in range(N):
            if i!=j:
                first_node_index.append(i)
                second_node_index.append(j)
                
                if is_molecule:
                    edge_type.append(bonds_dict[(i,j)] if i < j else bonds_dict[(j,i)])
                    
                elif is_structure:
                    site1 = struct.species[i]
                    site2 = struct.species[j]
                    try:
                        is_bond = CovalentBond.is_bonded(site1=site1, site2=site2)
                        
                        edge_type.append(bonds_type['single'] if is_bond else bonds_type['no-bond'])
                    except:
                        #catch no data errors
                        edge_type.append(bonds_type['no-data'])
                
                
    edge_index = torch.tensor([first_node_index, second_node_index], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds_type)).to(torch.float)
    
    return edge_index,edge_attr


def global_molecule_features(mol:Chem.rdchem.Mol):
    from rdkit.Chem import Descriptors, Descriptors3D

    ##BCUT and PartialCharge features are problematic for crystallographic derived molecules



    funcs_descriptor_2d = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 
               'qed', '_isCallable', '_descList', '_setupDescriptors', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 
               '_ChargeDescriptors', '_FingerprintDensity', 'FpDensityMorgan1',
               'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',
               'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 
               'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 
               'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 
               'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 
               'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 
               'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 
               'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
               'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 
               'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 
               'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 
               'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 
               'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
               'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 
               'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 
               'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
               'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 
               'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
               'fr_unbrch_alkane', 'fr_urea', 'AUTOCORR2D_1', 'AUTOCORR2D_2', 'AUTOCORR2D_3', 'AUTOCORR2D_4', 'AUTOCORR2D_5', 'AUTOCORR2D_6', 
               'AUTOCORR2D_7', 'AUTOCORR2D_8', 'AUTOCORR2D_9', 'AUTOCORR2D_10', 'AUTOCORR2D_11', 'AUTOCORR2D_12', 'AUTOCORR2D_13', 'AUTOCORR2D_14', 
               'AUTOCORR2D_15', 'AUTOCORR2D_16', 'AUTOCORR2D_17', 'AUTOCORR2D_18', 'AUTOCORR2D_19', 'AUTOCORR2D_20', 'AUTOCORR2D_21', 'AUTOCORR2D_22', 
               'AUTOCORR2D_23', 'AUTOCORR2D_24', 'AUTOCORR2D_25', 'AUTOCORR2D_26', 'AUTOCORR2D_27', 'AUTOCORR2D_28', 'AUTOCORR2D_29', 'AUTOCORR2D_30', 
               'AUTOCORR2D_31', 'AUTOCORR2D_32', 'AUTOCORR2D_33', 'AUTOCORR2D_34', 'AUTOCORR2D_35', 'AUTOCORR2D_36', 'AUTOCORR2D_37', 'AUTOCORR2D_38', 
               'AUTOCORR2D_39', 'AUTOCORR2D_40', 'AUTOCORR2D_41', 'AUTOCORR2D_42', 'AUTOCORR2D_43', 'AUTOCORR2D_44', 'AUTOCORR2D_45', 'AUTOCORR2D_46', 
               'AUTOCORR2D_47', 'AUTOCORR2D_48', 'AUTOCORR2D_49', 'AUTOCORR2D_50', 'AUTOCORR2D_51', 'AUTOCORR2D_52', 'AUTOCORR2D_53', 'AUTOCORR2D_54', 
               'AUTOCORR2D_55', 'AUTOCORR2D_56', 'AUTOCORR2D_57', 'AUTOCORR2D_58', 'AUTOCORR2D_59', 'AUTOCORR2D_60', 'AUTOCORR2D_61', 'AUTOCORR2D_62', 
               'AUTOCORR2D_63', 'AUTOCORR2D_64', 'AUTOCORR2D_65', 'AUTOCORR2D_66', 'AUTOCORR2D_67', 'AUTOCORR2D_68', 'AUTOCORR2D_69', 'AUTOCORR2D_70', 
               'AUTOCORR2D_71', 'AUTOCORR2D_72', 'AUTOCORR2D_73', 'AUTOCORR2D_74', 'AUTOCORR2D_75', 'AUTOCORR2D_76', 'AUTOCORR2D_77', 'AUTOCORR2D_78', 
               'AUTOCORR2D_79', 'AUTOCORR2D_80', 'AUTOCORR2D_81', 'AUTOCORR2D_82', 'AUTOCORR2D_83', 'AUTOCORR2D_84', 'AUTOCORR2D_85', 'AUTOCORR2D_86', 
               'AUTOCORR2D_87', 'AUTOCORR2D_88', 'AUTOCORR2D_89', 'AUTOCORR2D_90', 'AUTOCORR2D_91', 'AUTOCORR2D_92', 'AUTOCORR2D_93', 'AUTOCORR2D_94', 
               'AUTOCORR2D_95', 'AUTOCORR2D_96', 'AUTOCORR2D_97', 'AUTOCORR2D_98', 'AUTOCORR2D_99', 'AUTOCORR2D_100', 'AUTOCORR2D_101', 'AUTOCORR2D_102', 
               'AUTOCORR2D_103', 'AUTOCORR2D_104', 'AUTOCORR2D_105', 'AUTOCORR2D_106', 'AUTOCORR2D_107', 'AUTOCORR2D_108', 'AUTOCORR2D_109', 'AUTOCORR2D_110', 
               'AUTOCORR2D_111', 'AUTOCORR2D_112', 'AUTOCORR2D_113', 'AUTOCORR2D_114', 'AUTOCORR2D_115', 'AUTOCORR2D_116', 'AUTOCORR2D_117', 'AUTOCORR2D_118', 
               'AUTOCORR2D_119', 'AUTOCORR2D_120', 'AUTOCORR2D_121', 'AUTOCORR2D_122', 'AUTOCORR2D_123', 'AUTOCORR2D_124', 'AUTOCORR2D_125', 'AUTOCORR2D_126', 
               'AUTOCORR2D_127', 'AUTOCORR2D_128', 'AUTOCORR2D_129', 'AUTOCORR2D_130', 'AUTOCORR2D_131', 'AUTOCORR2D_132', 'AUTOCORR2D_133', 'AUTOCORR2D_134', 
               'AUTOCORR2D_135', 'AUTOCORR2D_136', 'AUTOCORR2D_137', 'AUTOCORR2D_138', 'AUTOCORR2D_139', 'AUTOCORR2D_140', 'AUTOCORR2D_141', 'AUTOCORR2D_142', 
               'AUTOCORR2D_143', 'AUTOCORR2D_144', 'AUTOCORR2D_145', 'AUTOCORR2D_146', 'AUTOCORR2D_147', 'AUTOCORR2D_148', 'AUTOCORR2D_149', 'AUTOCORR2D_150', 
               'AUTOCORR2D_151', 'AUTOCORR2D_152', 'AUTOCORR2D_153', 'AUTOCORR2D_154', 'AUTOCORR2D_155', 'AUTOCORR2D_156', 'AUTOCORR2D_157', 'AUTOCORR2D_158', 
               'AUTOCORR2D_159', 'AUTOCORR2D_160', 'AUTOCORR2D_161', 'AUTOCORR2D_162', 'AUTOCORR2D_163', 'AUTOCORR2D_164', 'AUTOCORR2D_165', 'AUTOCORR2D_166', 
               'AUTOCORR2D_167', 'AUTOCORR2D_168', 'AUTOCORR2D_169', 'AUTOCORR2D_170', 'AUTOCORR2D_171', 'AUTOCORR2D_172', 'AUTOCORR2D_173', 'AUTOCORR2D_174', 
               'AUTOCORR2D_175', 'AUTOCORR2D_176', 'AUTOCORR2D_177', 'AUTOCORR2D_178', 'AUTOCORR2D_179', 'AUTOCORR2D_180', 'AUTOCORR2D_181', 'AUTOCORR2D_182', 
               'AUTOCORR2D_183', 'AUTOCORR2D_184', 'AUTOCORR2D_185', 'AUTOCORR2D_186', 'AUTOCORR2D_187', 'AUTOCORR2D_188', 'AUTOCORR2D_189', 'AUTOCORR2D_190', 
               'AUTOCORR2D_191', 'AUTOCORR2D_192']

    funcs_descriptor_2d = [x for x in funcs_descriptor_2d if '_' not in x or str.index(x,'_') != 0]

    funcs_descriptor_3d = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'InertialShapeFactor', 'Eccentricity', 'Asphericity', 'SpherocityIndex']
    
    
    x=[]
    for func_name in funcs_descriptor_2d:
        func= Descriptors.__dict__[func_name]
        res = func(mol)
        x.append(res)

    for func_name in funcs_descriptor_3d:
        func = Descriptors3D.__dict__[func_name]
        x.append(func(mol))


    x= torch.tensor(x,dtype=torch.float)
    
    return x


