import getpass
import os
from pathlib import Path

LIBS_DIR = Path(os.path.join(os.getcwd(),'external'))

GRID_LENGTH = 200
GRID_CELL_SIZE = 0.01
# GRID_LENGTH = 400
# GRID_CELL_SIZE = 0.005
STEP_FREQUENCY = 88200
NUM_SUBSTEPS = 2
IR_POSITION = [0, 0.7, .2]
SAMPLE_POSITION = [0, -0.7, 0.2]
# IR_POSITION = [0.9, 0, -0.6]
# SAMPLE_POSITION = [0.9, 0, -0.6]
N_STEPS = 8820

# VGGISH
PSEUDO_SAMPLE_RATE = 16000