import pandas as pd
from typing import Optional, Callable
import os
import os.path as osp
from typing import Tuple, List, Dict, Union



from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

from features import edge_features, node_features, pymatgen_node_features

## removing warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


import torch
import torch.nn.functional as F
from tqdm import tqdm


from torch_geometric.data import (
    Data)

from pymatgen.core import Structure, Element, Molecule
from pymatgen.io.babel import BabelMolAdaptor


import ssl
import sys

from urllib.request import Request, urlopen



def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder,exist_ok=True)

    context = ssl.create_default_context()
    ## precising agent to pass by forbidden access for some datasets (matbench)
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
 
    data = urlopen(req) 
    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path




def download_dataset(raw_dir:str, filename:str, raw_url:str, target_columns:List[str], data_column_name: str):
    """_summary_

    Args:
        raw_dir (str): _description_
        filename (str): _description_
        raw_url (str): _description_
        target_columns (List[str]): _description_
        data_column_name (str): _description_
    """
    complete_path = f"{raw_dir}/{filename}"
    filepath= download_url(raw_url, raw_dir)
    
    ## if original dataset was in csv, write it back in csv after filtering
    ## for json.gz dataset for materials, we will have to take care of it later downstream 
    if "csv" in filename:
    
        df = pd.read_csv(complete_path)
        col_list=[data_column_name]+ target_columns
        df.drop(df.columns.difference(col_list), axis=1, inplace=True)
        df.set_index(data_column_name, drop=True, inplace=True)
        df.to_csv(complete_path, encoding="utf-8")
    
    return 



def from_smiles_to_molecule_and_coordinates(smile: str, seed:int, add_hydrogen: bool) ->  Tuple[Chem.rdchem.Mol, torch.Tensor]:
    """From smiles string, generate 3D coordinates using seeded ETKDG procedure.
    Args:
        smile (str): valid smiles tring of molecule
    Returns:
        Tuple[Chem.rdchem.Mol, torch.Tensor]: molecule and 3D coordinates of atom
    """
    
    try:
        m = Chem.MolFromSmiles(smile)
    except:
        print("invalid smiles string")
        return None, None
    
    # necessary to add hydrogen for consistent conformer generation
    try:
        m = Chem.AddHs(m)
    except:
        print("can't add hydrogen")
        return None, None
    
    ## 3D conformer generation
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = seed
    #ps.coordMap = coordMap = {0:[0,0,0]}
    err = AllChem.EmbedMolecule(m,ps)
    
    # conformer generation failed for some reason (molecule too big is an option)
    if err !=0:
        return None, None
    conf = m.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)
    ## if we dont want hydrogen, we need to rebuild a molecule without explicit hydrogens
    if not(add_hydrogen):
        # https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/811862b82ce7402b8ba01201a5d0334a%40uni.lu/
        params = Chem.RemoveHsParameters()
        params.removeDegreeZero = True
        m = Chem.RemoveHs(m, params)
        
    return m, pos    



def from_structure_to_molecule(struct:Structure) -> Chem.rdchem.Mol:

    # we will compute distances directly using the pymatgen structure

    #then the following conversion : pymatgen.Structure -> pymatgen.Molecule -> pybel_mol -> mol file (to retain 3D information) ->  rdkit molecule
    mol = Molecule(species=struct.species, coords=struct.cart_coords)
    try:
        adaptor = BabelMolAdaptor(mol).pybel_mol
    except:
        print("unable to convert from pymatgen structure")
        return None
    
    try:
        #ideally, we would like to give the correct 3D coordinates to the molecule, so we use .mol file
        mol_file = adaptor.write('mol')

        new_mol = Chem.MolFromMolBlock(mol_file, sanitize=False)
        problems = Chem.DetectChemistryProblems(new_mol)
        len_problems=len(problems)

        if len_problems > 0:
            return None
    except:
        print("unable to convert to rdkit molecule")
        return None
    
    return new_mol






def from_molecule_to_graph(mol:Chem.rdchem.Mol, y:torch.Tensor, pos:torch.Tensor, name:str, idx:int, data: Union[str,Structure]) -> Data:
    
    x,z =  pymatgen_node_features(mol=mol)
    
    edge_index, edge_attr = edge_features(mol=mol)
    
    #x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    
    # 5. Bundling everything into the Data (graph) type

    #x = torch.cat([x1.to(torch.float), x2], dim=-1)
    
    if isinstance(data, str):
        graph = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, y=y, name=name, idx=idx)
    elif isinstance(data, Structure):
        (row, col) = edge_index
        ## getting distances from distance matrix that is aware of mirror images
        distance_matrix = data.distance_matrix
        dist = torch.tensor(distance_matrix[row,col],dtype=torch.float)
        if dist is None or distance_matrix is None:
            return None
        else:
            graph= Data(x=x, z=z,edge_index=edge_index, edge_attr=edge_attr, y=y, name=name, idx=idx, dist=dist)
            
    if isinstance(data,Structure) and not(hasattr(graph, 'dist')):
        return None

    
    return graph
    


def data_to_graph(data:Union[str,Structure], y:torch.Tensor, idx: int, seed:int, add_hydrogen:bool, pre_transform: Callable, pre_filter:Callable) -> Data:
    
    
    if isinstance(data,str):
        name= data
        ## 2. ETKDG seeded method 3D coordinate generation
        mol, pos = from_smiles_to_molecule_and_coordinates(smile=data, seed=seed, add_hydrogen=add_hydrogen)
        
    elif isinstance(data,Structure):
        
        if add_hydrogen:
            raise ValueError("Explicit hydrogens is not yet supported for crystallographic data")
        name = data.formula
        ## no need for 3D coordinate generation as crystallographic structure is given
        mol = from_structure_to_molecule(struct=data)
        ## we still initialize pos for compatibility with functions
        pos = None
    
    if mol is None:
        return None
    
    graph= from_molecule_to_graph(mol=mol, y=y, pos=pos, name=name, idx=idx, data=data)
    
    if pre_filter is not None and not pre_filter(graph):
        return None
    
    if pre_transform is not None:
        graph = pre_transform(graph)
        
    return graph    


