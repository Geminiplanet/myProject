CHAR_LIST = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti", "V",
             "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Br", "Sr", "Zr", "Mo", "Pd", "Yb", "Ag", "Cd", "Sb",
             "I", "Ba", "Nd", "Gd", "Dy", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
             "n", "c", "o", "s", "se",
             "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "(", ")", "[", "]",
             "-", "=", "#", "/", "\\", "+", "@", "<", ">", "."]
TOX21_TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
               'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
MAX_SEQ_LEN = 200

BATCH = 32
SUBSET = 1000
ADDENDUM = 200

CYCLES = 5

EPOCHP = 100
EPOCH = 200
EPOCHV = 100  # VAAL number of epochs
MILESTONES = [160, 240]
CUDA_VISIBLE_DEVICES = 0
WDECAY = 5e-4#2e-3# 5e-4