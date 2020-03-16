# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:50:18 2020

@author: Khalil
"""


def insert_i(in_list, n_insert, n_deposit, outputs = [], depth = 0, shift = 0):
    
    depth += 1

    for k in range(1 + shift, n_insert + n_deposit):
        
        print(in_list) 
        print(k)       
        temp = in_list.copy()
        temp.insert(k, i)
        out = temp.copy()
        
        if depth == n_insert:
            outputs += [out]
        else:
            insert_i(out, n_insert, n_deposit, outputs, depth, k)
        
    return outputs
    
        
_max = 6
in_list = [1,0,2]
i = -1
s = len(in_list)

#outputs = []
#
#n_insert = 3
#
#for n in range(1,_max - s + n_insert):
#    out_list = in_list.copy()
#    out_list.insert(n, i)
#    temp1 = out_list.copy()
#    print(n)
#    for m in range(n+1,_max - s + n_insert):
#        temp2 = temp1.copy()
#        temp2.insert(m, i)
#        temp3 = temp2.copy()
#        print("m: %i" %m)
#        for k in range(m+1,_max - s + n_insert):
#            temp4 = temp3.copy()
#            temp4.insert(k, i)
#            temp5 = temp4.copy()
#            print("k: %i" %k)
#            
#            outputs += [temp5]
            
#            for l in range(k+1,_max - s + n_insert):
#                temp6 = temp5.copy()
#                temp6.insert(l, i)
#                temp7 = temp6.copy()
#        
#                outputs += [temp7]

#
outputs = insert_i(in_list,_max-s,s)
