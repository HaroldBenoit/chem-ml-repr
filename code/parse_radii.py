from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import torch

with open("atomic_radii.html") as fp:
    soup = BeautifulSoup(fp, "html.parser")
    column_names= ["atomic_number", "element", "single_bond_1", "single_bond_2", "double_bond", "triple_bond"]
    dataframe = {name:[] for name in column_names}
    for row in soup.tbody.find_all("tr"):
        for i, data in enumerate(row.findAll("td")):
            #print(data["data-th"], data.text)
            text = data.text
            if text == "-":
                text = np.nan
                
            dataframe[column_names[i]].append(text)
            
df = pd.DataFrame(dataframe)
dtype={"atomic_number":'int32',
                                "element":str,
                                "single_bond_1":'Int64',
                                "single_bond_2":'Int64',
                                "double_bond":'Int64',
                                "triple_bond":'Int64'}

df = df.astype(dtype)

df.to_csv("../important_data/atomic_radius.csv")

## creating the official mapping from atomic to atomic radius, unit is Angstrom
atom_number_to_radius=dict(zip(list(df.atomic_number), list(df.single_bond_2/100)))

torch.save(atom_number_to_radius, "../important_data/atom_number_to_covalent_radius.pt")