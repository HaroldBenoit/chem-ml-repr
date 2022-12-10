from datasets_classes import MatBenchMpIsMetal
from datasets_classes import QM9Dataset
from tqdm import tqdm

dataset = MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")

#datatset =QM9Dataset(root="../data/qm9")

for i in tqdm(range(len(dataset))):
    data = dataset[i]
    
    if not(hasattr(data,'dist')):
        print("Bad index ",i)
        
