from .data import *
from .molecules import *
from .similarity import *
from .queue import *
from .fingerprints import *
from .train import *

def create_path(output_path):
    if not output_path.exists():
        output_path.mkdir(parents=True)
