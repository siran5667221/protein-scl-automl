import traceback
import os
from pathlib import Path
import iFeatureOmegaCLI

def process_feature_type(ft: str):
    print(f"[RUN] {ft}")
    try:
        fasta_path = os.environ.get("IFEATURE_FASTA")
        if not fasta_path:
            raise ValueError("Set IFEATURE_FASTA to the intended input FASTA.")
        parameter_path = os.environ.get(
            "IFEATURE_PARAMETERS",
            str(Path(__file__).resolve().parents[2] / "parameters" / "Protein_parameters_setting.json")
        )
        protein = iFeatureOmegaCLI.iProtein(fasta_path)
        protein.import_parameters(parameter_path)
        protein.get_descriptor(ft)

        df = getattr(protein, "encodings", None)
        err_msg = getattr(protein, "error_msg", None)

        if df is None:
            msg = err_msg or "encodings is None (unknown error)"
            print(f"[ERR] {ft}: {msg}")
            return ft, None, msg
        else:
            print(f"[OK ] {ft}: shape={getattr(df, 'shape', None)}")
            return ft, df, None

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[EXC] {ft}: {e}\n{tb}")
        return ft, None, f"{e}\n{tb}"
