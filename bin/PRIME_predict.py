"""
PRIME must be installed and executable from the command line.
source: https://github.com/GfellerLab/PRIME
"""
import argparse
import sys
import os
import pandas as pd
from tqdm import tqdm


class PRIME_prediction:
    def __init__(self, input, hla, pep, label, out):
        os.system("mkdir -p .tmp")
        
        pep_df = pd.read_csv(input)

        def convert_hla_format(hla_string):
            # Remove the 'HLA-' part and replace '*' with nothing, ':' with nothing
            return hla_string.replace('HLA-', '').replace('*', '').replace(':', '')

        # Applying the function to the DataFrame
        pep_df['nMHC'] = pep_df[hla].apply(convert_hla_format)
        pep_df["mergeHLA_pep"] = pep_df[pep] + "_" + pep_df['nMHC']
        pep_df = pep_df[pep_df[pep].str.len().between(8, 14)]
        final_df_list = []

        pep_get_group = pep_df.groupby('nMHC')
        for y in tqdm(pep_df["nMHC"].unique()):
            groupbyHLA_df = pep_get_group.get_group(y)
            BA_predict = self.prime_pred(groupbyHLA_df[pep], y)
            merged1 = groupbyHLA_df.merge(BA_predict, how = "inner", left_on = "mergeHLA_pep", right_on = "mergeHLA_pep", suffixes=('', '__3')).drop_duplicates().reset_index(drop=True)
            #flurry_predict = self.mhcFlurry_process(merged1[pep], y)
            #merged3 = merged2.merge(flurry_predict, how = "inner", left_on = "mergeHLA_pep", right_on = 'mergeHLA_pep', suffixes=('', '__2')).drop_duplicates().reset_index(drop=True)

            final_df_list.append(merged1)

        tmp_df = pd.concat(final_df_list, ignore_index=True)
        tmp_df = tmp_df.drop_duplicates(subset=["mergeHLA_pep"])
        final_df = pep_df.merge(tmp_df, how = "inner", left_on = "mergeHLA_pep", right_on = "mergeHLA_pep", suffixes=('', '__2')).drop_duplicates().reset_index(drop=True)
        #final_df.to_csv()
        final_df = pd.DataFrame({"Peptide": final_df["Peptide"], "MHC": final_df[hla], "Label": final_df[label],
                           "PRIME_score": final_df["PRIME_score"], "PRIME_rank": final_df["PRIME_rank"]})
        final_df = final_df.reset_index(drop=True)
        final_df.to_csv(out, index = False)

        pass

    def check_arg(args=None):
        parser = argparse.ArgumentParser(description='Script to predict immunogenicity of input peptides with PRIME 2.0.')

        parser.add_argument(	'--input' ,type=str,
                    help='input csv file')

        parser.add_argument(	'--hla' ,type=str,
                    help='column name of HLA')

        parser.add_argument(	'--pep' ,type=str,
                    help='column name of peptide')

        parser.add_argument(	'--l' ,type=str,
                    default="1",
                    help='label (1 or 0)')

        parser.add_argument(	'--out' ,type=str,
                    default="PRIME_out.csv",
                    help='output file')

        args = parser.parse_args(args)

        return args

    def prime_pred(self, pep, hla):
        ba_tmp_file_name = os.path.join(os.getcwd(), ".tmp/tmp_pep_ba.txt")
        pep.to_csv(ba_tmp_file_name, sep="\t", header=None, index=False)
        cmd = "PRIME2 -i {} -a {} -o .tmp/prime.pred".format(ba_tmp_file_name, hla)
        os.system(cmd)
        pred_rs = self.prime_output_process(".tmp/prime.pred", hla)
        return pred_rs

    def prime_output_process(self, input_path, hla):
        df = pd.read_csv(input_path, sep="\t", header=0, comment="#")
        concat_df = pd.DataFrame({"Peptide": df["Peptide"], "nMHC": hla, "PRIME_score": df.iloc[:,6], "PRIME_rank": df.iloc[:,5]})
        concat_df["mergeHLA_pep"] = concat_df["Peptide"] + "_" + hla
        concat_df.drop_duplicates(subset=['mergeHLA_pep'])
        return concat_df

if __name__ == '__main__':
    args = PRIME_prediction.check_arg(args = sys.argv[1:])
    PRIME_prediction(args.input, args.hla, args.pep, args.l, args.out)
