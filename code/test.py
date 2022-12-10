from datasets_classes import MatBenchMpIsMetal


dataset = MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")

for i in range(len(dataset)):
    data = dataset[i]
    
    if not(hasattr(data,'dist')):
        print("Bad index ",i)
        
