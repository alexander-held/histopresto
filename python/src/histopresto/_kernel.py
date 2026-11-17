"""ctypes interface to the C kernel (kernel.c, built by CMake when the package is installed)."""

import ctypes
import importlib.resources
import sys

LIBRARY = "libkernel.dylib" if sys.platform == "darwin" else "libkernel.so"


I32 = ctypes.c_int32
F64 = ctypes.c_double
PTR = ctypes.c_void_p

# (name, numpy dtype) of the arrays in hf_model, in the order of the C struct
ARRAYS = [
    ("blk_offset", "i4"), ("blk_n", "i4"), ("blk_cell", "i4"),
    ("nominal", "f8"),
    ("scal_start", "i4"), ("scal_count", "i4"), ("scal_par", "i4"),
    ("ns_start", "i4"), ("ns_count", "i4"), ("ns_par", "i4"),
    ("ns_lnhi", "f8"), ("ns_lnlo", "f8"), ("ns_coef", "f8"),
    ("hs_start", "i4"), ("hs_count", "i4"), ("hs_par", "i4"), ("hs_data", "i4"),
    ("hs_up", "f8"), ("hs_dn", "f8"),
    ("gam_start", "i4"), ("gam_count", "i4"), ("gam_off", "i4"), ("gam_idx", "i4"),
    ("gauss_idx", "i4"), ("gauss_aux", "i4"), ("gauss_sigma", "f8"),
    ("pois_idx", "i4"), ("pois_aux", "i4"), ("pois_tau", "f8"),
    ("nu", "f8"), ("cell_nu", "f8"), ("cell_base", "f8"), ("cell_gamma", "f8"),
    ("block_factor", "f8"), ("cell_weight", "f8"), ("ns_val", "f8"), ("ns_der", "f8"),
    ("fn_start", "i4"), ("fn_count", "i4"), ("fn_slot", "i4"),
    ("tape_op", "i4"), ("tape_a", "i4"), ("tape_b", "i4"), ("tape_value", "f8"),
    ("slot_val", "f8"), ("slot_adj", "f8"),
]
COUNTS = ["n_bins", "n_pars", "n_blocks", "n_cells", "n_gauss", "n_pois", "n_slots"]


class HFModel(ctypes.Structure):
    _fields_ = [(name, I32) for name in COUNTS] + [(name, PTR) for name, _ in ARRAYS]


with importlib.resources.as_file(importlib.resources.files(__package__) / LIBRARY) as _path:
    lib = ctypes.CDLL(str(_path))

lib.hf_evaluate.argtypes = [PTR, PTR]
lib.hf_evaluate.restype = None
lib.hf_accumulate.argtypes = [PTR, PTR, PTR, PTR, ctypes.c_int]
lib.hf_accumulate.restype = None
lib.hf_constant.argtypes = [PTR, PTR]
lib.hf_constant.restype = F64
lib.hf_nll.argtypes = [PTR, PTR, F64, PTR, PTR]
lib.hf_nll.restype = F64
lib.hf_minimize.argtypes = [PTR, PTR, F64, PTR, PTR, PTR, PTR, ctypes.c_int, PTR, PTR, F64, ctypes.c_int, PTR, PTR, PTR, PTR, PTR, PTR, PTR]
lib.hf_minimize.restype = ctypes.c_int
lib.hf_hessian.argtypes = [PTR, PTR, F64, PTR, PTR, ctypes.c_int, PTR, ctypes.c_int, PTR]
lib.hf_hessian.restype = None
