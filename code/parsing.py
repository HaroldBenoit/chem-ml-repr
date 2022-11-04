from typing import Any, Dict, TextIO


def parse_cif_file(cif_file:TextIO) -> Dict[Dict[Dict[str, Any]]]:

    """Given a text file containing a list of CIF molecule descriptions (simply appended one next to the other),
    parses CIF structurs into a dictionary. The molecule names are given in SMILES notation.

    Returns:
        _type_: dictionary with the following structure

            {
        "molecule_name":{
            "atom_name0":{
                "atom_name":str,
                "fract_x":float,
                "fract_y":float,
                "fract_z":float,
                "fract_x":float,
                "site_occupancy":float
                },
            "atom_name1":{...},
            ...
        }
        ...
    }
    """

    ## dictionary of dictionaries containing complete 3D description for each molecule in the CIF file
    mol_3d_dic = {}
    curr_name=""
    i = 0

    ## strings contained in lines we wish to ignore
    to_ignore=["data_I", "loop", "atom", "openbabel"]

    with open(cif_file, "r") as f:
        new_line = f.readline()
        while(new_line !=  "" and i < 30):
            #start of a new molecule
            if new_line.__contains__("_chemical_name"):
                ## split by whitespace and strip of all characters
                split = list(map(str.strip,new_line.split(" ")))
                ## second element is the name and you need to remove the outer quotes ''
                curr_name= split[1][1:-1]
                mol_3d_dic[curr_name] = {}
                print(f"curr_name: {curr_name}")

                # doesn't contain words to ignore and is not only whitespace
            elif not(any(ele in new_line for ele in to_ignore) or new_line.isspace()):
                # we split by whitespace but we're gonna get a lot empty strings so we remove them
                elems = [ele.strip() for ele in new_line.split(" ") if not(len(ele) == 0) ]
                curr_molecule_dic = mol_3d_dic[curr_name]
                atom_numbering = elems[0]
                curr_molecule_dic[atom_numbering] = {}
                curr_atom_dic = curr_molecule_dic[atom_numbering]
                curr_atom_dic["atom_name"] = elems[1]
                curr_atom_dic["fract_x"] = float(elems[2])
                curr_atom_dic["fract_y"] = float(elems[3])
                curr_atom_dic["fract_z"] = float(elems[4])
                curr_atom_dic["site_occupancy"] = float(elems[5])
                #print(f"new atom: {elems}")


            new_line = f.readline()
            i+=1


    return mol_3d_dic
