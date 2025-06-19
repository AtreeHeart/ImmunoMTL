"""
NetMHCpan must be installed and executable from the command line.
source: https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
"""
import argparse
import sys
import os
import pandas as pd
from tqdm import tqdm
from TNinputcheck import checkinput

class BABS_prediction:
    def __init__(self, input, hla, pep, label, out):
        # Create a temporary directory to store intermediate files
        os.system("mkdir -p .tmp")
        
        # Load the input peptide data from a CSV file
        pep_df = pd.read_csv(input)
        
        # Clean the HLA data by removing the '*' characters
        pep_df['nMHC'] = pep_df[hla].str.replace('*','')
        
        # Validate peptide sequences to ensure they are formatted correctly
        invalidpep = checkinput.validate_peptide_sequence(pep_df[pep])
        pep_df = pep_df[~pep_df[pep].isin(invalidpep)]
        
        pep_df["mergeHLA_pep"] = pep_df[pep] + "_" + pep_df['nMHC']
        final_df_list = []
        
        # Group peptides by HLA type and predict binding affinity for each group
        pep_get_group = pep_df.groupby('nMHC')
        for hla_type in tqdm(pep_df["nMHC"].unique()):
            groupbyHLA_df = pep_get_group.get_group(hla_type)
            BA_predict = self.netBA_pred(groupbyHLA_df[pep], hla_type)
            merged_df = groupbyHLA_df.merge(BA_predict, how="inner", left_on="mergeHLA_pep", right_on="mergeHLA_pep", suffixes=('', '__2')).drop_duplicates().reset_index(drop=True)
            final_df_list.append(merged_df)

        # Concatenate all results and remove duplicates
        final_df = pd.concat(final_df_list, ignore_index=True)
        final_df["mergeHLA_pep"] = final_df["Peptide"] + "_" + final_df["HLA"]
        final_df = final_df.drop_duplicates(subset=["mergeHLA_pep"])
        
        # Select and organize the final columns for output
        final_df = pd.DataFrame({
            "Peptide": final_df["Peptide"], 
            "MHC": final_df[hla], 
            "Label": final_df[label],
            "EL-score": final_df["EL_Score"], 
            "EL_Rank": final_df["EL_Rank"]
        })
        
        final_df = final_df.reset_index(drop=True)
        final_df.to_csv(out, index=False)

    def check_arg(args=None):
        parser = argparse.ArgumentParser(description='Script to predict binding affinity of input peptides.')

        parser.add_argument('--input', type=str, help='input CSV file containing peptides and HLA types')
        parser.add_argument('--hla', type=str, help='column name of HLA in the input file')
        parser.add_argument('--pep', type=str, help='column name of peptide sequences in the input file')
        parser.add_argument('--l', type=str, default="1", help='label column name (1 or 0)')
        parser.add_argument('--out', type=str, default="BA_prediction_output.csv", help='output CSV file for predictions')

        args = parser.parse_args(args)
        return args

    def netBA_pred(self, pep, hla):
        ba_tmp_file_name = os.path.join(os.getcwd(), ".tmp/tmp_pep_ba.txt")
        pep.to_csv(ba_tmp_file_name, sep="\t", header=None, index=False)
        
        # Run netMHCpan to predict binding affinity
        cmd = "netMHCpan -p {} -BA -a {} -xls -xlsfile .tmp/ba.pred.xls".format(ba_tmp_file_name, hla)
        os.system(cmd)
        
        # Process the netMHCpan output file and return the results
        pred_rs = self.netMHC_output_process(".tmp/ba.pred.xls", "BA", hla)
        return pred_rs

    def netMHC_output_process(self, input_path, mode, hla):
        df = pd.read_csv(input_path, sep="\t", skiprows=1, header=None, index_col=False)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        concat_df = pd.DataFrame({
            "Peptide": df["Peptide"], 
            "HLA": hla, 
            "core": df["core"],
            "icore": df["icore"], 
            "EL_Score": df["EL-score"], 
            "EL_Rank": df["EL_Rank"]
        })
        
        concat_df["mergeHLA_pep"] = concat_df["Peptide"] + "_" + concat_df["HLA"]
        concat_df.drop_duplicates(subset=['mergeHLA_pep'])
        return concat_df


if __name__ == '__main__':
    args = BABS_prediction.check_arg(args=sys.argv[1:])
    BABS_prediction(args.input, args.hla, args.pep, args.l, args.out)