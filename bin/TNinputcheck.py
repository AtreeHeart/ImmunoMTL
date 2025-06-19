import warnings
import pandas as pd
import os, sys
sys.path.append('../')

class checkinput:
    def __init__(self):
        pass

    def validate_peptide_sequence(peplist):
        valid_characters = set("ACDEFGHIKLMNPQRSTVWY")

        rmpep = []
        for peptide_sequence in peplist:
            length = len(peptide_sequence)

            if not set(peptide_sequence).issubset(valid_characters):
                rmpep.append(peptide_sequence)
                warnings.warn("Peptide sequence contains invalid amino acid characters.")

            if length < 8 or length > 11:
                rmpep.append(peptide_sequence)
                warnings.warn("Peptide sequence must be between 8-11 amino acids in length.")

        return rmpep
    
    def validate_mhc_support(hlalist):
        absolute_path = os.path.realpath(__file__)
        parent_dir = "/".join(absolute_path.split("/")[:-1])

        HLA_ava = pd.read_csv(parent_dir + "../HLA/IMMpred_supported_hla.csv")
        HLA_ava['nMHC'] = HLA_ava["HLA"].str.replace('*','')
        rmhla = []

        for i, hla in enumerate(hlalist):
            if '*' in hla:
                if hla not in HLA_ava["HLA"].values:
                    raise ValueError("detect unsupported HLA type: " + hla + ", which is at line " + str(i))
                    rmhla.append(hla)
            else: 
                if hla not in HLA_ava["nMHC"].values:
                    raise ValueError("detect unsupported HLA type: " + hla + ", which is at line " + str(i))
                    rmhla.append(hla)
        return rmhla
