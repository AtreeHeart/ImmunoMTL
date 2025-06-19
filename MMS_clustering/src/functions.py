import math
import numpy as np

def entropy(df):
    res = []
    for i in range(len(df.columns)-2):
        AA_unique = df[i+1].unique()
        entropy_ls = []
        for y in range(len(AA_unique)):
            p = df[i+1].tolist().count(AA_unique[y])/len(df[i+1])
            entropy_ls.append( p * math.log(p,2))
        #entropy_sum = -1 * sum(entropy_ls)
        entropy_sum = math.log(20, 2)- (-1*sum(entropy_ls) + (1/np.log(2)*(20-1)/(2*len(df[i+1]))))
        
        res.append(entropy_sum)    
    return res

def normalized(lst):
    norm_lst = []
    for i in lst:
        norm_lst.append(i/sum(lst))
    return norm_lst