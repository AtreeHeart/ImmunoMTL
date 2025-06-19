# === Development Environment ===
"""
Python version:       3.10.15 
pandas version:       2.2.3
numpy version:        1.26.4
tqem version: 		  4.67.1
biopython version:    1.85
mhcflurry version:    2.0.0

NetMHCpan version must be installed and executable from the command line.
source: https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
"""

# === Imports ===
import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import subprocess
from tqdm import tqdm
from Bio import SeqIO
from mhcflurry import Class1PresentationPredictor
import gc

class Npep_module:

	def __init__(self, hla_input):
		hla_list = pd.read_csv(hla_input)
		new_hla_list = []
		for hla in hla_list["HLA"]:
			new_hla_list.append(hla.replace('*', ''))
		self.output_obj = self.main(new_hla_list)
		gc.collect()
		pass

	def check_arg(args=None):
		parser = argparse.ArgumentParser(description='Script to predict binding affinity and stability of human noraml proteome (from uniprot). example: python3 Npep_generator2.py --hla ../HLA_clustering/hla/clustering_res.csv')
		parser.add_argument(	'--hla' ,type=str,
					help='ex. csv file.')

		args = parser.parse_args(args)

		return args
	def mhcflurry_pred(self, prot_dict, hla):
		final_df = pd.DataFrame()
		predictor = Class1PresentationPredictor.load()

		for i in tqdm(range(len(prot_dict)//1000+1)):
			# activate prediction (mhcFlurry)
			if i == 0:
				prot_dict_split = {k: prot_dict[k] for k in list(prot_dict)[:1000]}
			elif i != 0 & (i+1)*1000 < len(prot_dict):
				prot_dict_split = {k: prot_dict[k] for k in list(prot_dict)[i*1000+1:(i+1)*1000]}
			else:
				prot_dict_split = {k: prot_dict[k] for k in list(prot_dict)[i*1000:len(prot_dict)]}
			res = predictor.predict_sequences(
				sequences= prot_dict_split,
				alleles={"sample 1": hla},
				peptide_lengths = [8,9,10,11],
				result = 'all', 
				include_affinity_percentile = False,
				verbose=0, throw=False)

			df = res[res["presentation_percentile"] <= 2]
			df.rename(columns = {"peptide":"Peptide"}, inplace = True)
		final_df = pd.concat([final_df, df])

		return final_df

	def BA_pred(self, pep, hla):
		ba_tmp_file_name = os.path.join(os.getcwd(), ".tmp_files/tmp" + "_pep_ba.txt")
		pep.to_csv(ba_tmp_file_name, sep="\t", header=None, index=False)
		cmd = "netMHCpan -p {} -a {}  -xls -xlsfile .tmp_files/ba.pred.xls".format(ba_tmp_file_name, hla)
		subprocess.run(cmd, shell=True, stdout=None)
		pred_rs = self.netMHC_output_process(".tmp_files/ba.pred.xls", "BA", hla)

		return pred_rs

	def netMHC_output_process(self, input_path, mode, hla):
		df = pd.read_csv(input_path, sep="\t", skiprows=1, index_col=False)

		if mode == "BA":
			df = df.drop(['Pos','ID', 'Ave','NB'], axis=1)
			concat_df = pd.DataFrame({"Peptide": df["Peptide"], "HLA": hla, "core": df.iloc[:,1], 
				"icore": df.iloc[:,2], "EL_Score": df.iloc[:,3], "EL_Rank": df.iloc[:,4]})
			concat_df = concat_df[pd.to_numeric(concat_df['EL_Rank']) <= 2]
			pass
		elif mode == "BS":
			df = df.drop(['Pos','ID','Ave','NB'], axis=1)
			concat_df = pd.DataFrame({"Peptide": df["Peptide"], "HLA": hla, "Pred": df.iloc[:,1], 
				"Thalf": df.iloc[:,2],"Rank": df.iloc[:,3]})
			concat_df = concat_df[pd.to_numeric(concat_df['Rank']) <= 2]
		concat_df.drop_duplicates(subset=['Peptide'])
		return concat_df

	def main(self, hla):
		import random
		# Make tmp folder
		os.system("mkdir -p .tmp_files")
		final_matrix = pd.DataFrame()

		# === Load and Preprocess Data ===
		# Human Proteomes .fasta can be obtained from Uniprot
		# Source: https://www.uniprot.org/proteomes/UP000005640
		# Note: This dataset is not included in this github repo

		fasta_path = "../data/uniport_all_prote_20oct2022.fasta"
		if not os.path.exists(fasta_path):
			print(f"[ERROR] Required FASTA file not found at: {fasta_path}")
			print("Please download the human proteome FASTA file from UniProt:")
			print("→ https://www.uniprot.org/proteomes/UP000005640")
			print("And place it at '../data/uniport_all_prote_20oct2022.fasta'")
			sys.exit(1)

		record_dict = SeqIO.index(fasta_path, "fasta")

		# Only keep sequences with standard amino acids
		standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
		prot_dict_all = {k: str(v.seq) for k, v in record_dict.items() if set(str(v.seq)).issubset(standard_aas)}

		# Randomly sample 100 proteins with fixed seed
		random.seed(42)
		sampled_keys = random.sample(list(prot_dict_all.keys()), 100)
		prot_dict = {k: prot_dict_all[k] for k in sampled_keys}

		print(f"Total proteins selected: {len(prot_dict)}")

		pred_ba_flurry_matrix = self.mhcflurry_pred(prot_dict, hla)
		lens = []
		for pep in pred_ba_flurry_matrix["Peptide"]:
			lens.append(len(pep))
		pred_ba_flurry_matrix["Len"] = lens
		print(pred_ba_flurry_matrix)
		pred_ba_flurry_matrix.to_csv("testout.csv")
		gp = pred_ba_flurry_matrix.groupby("best_allele")

		for hla_ in pred_ba_flurry_matrix["best_allele"].unique():
			hla_df = pred_ba_flurry_matrix[pred_ba_flurry_matrix["best_allele"] == hla_]
			for i in range(8,12):

				if len(hla_df[hla_df["Len"] == i]["Peptide"]!= 0 ):

					#run netMHCpan prediction
					pred_ba_matrix = self.BA_pred(hla_df[hla_df["Len"] == i]["Peptide"], hla_)

					#merge by intersection
					ba_merge = pd.merge(pred_ba_matrix, hla_df[hla_df["Len"] == i], how='inner', on='Peptide', suffixes=('', '_drop'))
					ba_merge.drop([col for col in ba_merge.columns if 'drop' in col], axis=1, inplace=True)

					#final merge
					ba_merge.drop([col for col in ba_merge.columns if 'drop' in col], axis=1, inplace=True)
					final_matrix = pd.concat([final_matrix, ba_merge])

			final_matrix = final_matrix.drop_duplicates(subset=["Peptide", "HLA"])
			os.system("mkdir -p ../data/HN_pepbyHLA")
			prefix = "../data/HN_pepbyHLA/" + hla_ + "_H_N.csv"
			final_matrix.to_csv(prefix)

		pass

if __name__ == '__main__':
	args = Npep_module.check_arg(args = sys.argv[1:])
	args_var = vars(args)
	Npep_module(args_var["hla"])
