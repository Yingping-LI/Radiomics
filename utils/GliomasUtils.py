#!/usr/bin/env python
# coding: utf-8

"""
Some utils for gliomas dataset.
"""


#================= Functions to convert three binary labels to one multi-class label. ===============================
def caculate_tumor_subtype(data):
    """
    Define the tumor type according to the different combinations of tumor grade, IDH mutant and 1p/19q codeleted status.
    """
    
    if data["is_GBM"]==0 and data["is_IDH_mutant"]==1 and data["is_1p19q_codeleted"]==1:
        tumor_subtype_description="LGG, IDH mutant, 1p/19q codeleted"
        tumor_subtype=1
        
    elif data["is_GBM"]==0 and data["is_IDH_mutant"]==1 and data["is_1p19q_codeleted"]==0:
        tumor_subtype_description="LGG, IDH mutant, 1p/19q non-codeleted"
        tumor_subtype=2
        
    elif data["is_GBM"]==0 and data["is_IDH_mutant"]==0:
        tumor_subtype_description="LGG, IDH wildtype"
        tumor_subtype=3
        
    elif data["is_GBM"]==1 and data["is_IDH_mutant"]==1:
        tumor_subtype_description="GBM, IDH mutant"  
        tumor_subtype=4
        
    elif data["is_GBM"]==1 and data["is_IDH_mutant"]==0:
        tumor_subtype_description="GBM, IDH wildtype"
        tumor_subtype=5
    
    return tumor_subtype_description, tumor_subtype

def get_tumor_subtype_description(data):
    """
    Used to add new columns to describe the tumor_subtype by words;
    """
    tumor_subtype_description, tumor_subtype=caculate_tumor_subtype(data)
    
    return tumor_subtype_description

def get_tumor_subtype(data):
    """
    Used to add new columns to describe the tumor_subtype by number in {1,2,3,4,5}.
    """
    tumor_subtype_description, tumor_subtype=caculate_tumor_subtype(data)
    
    return tumor_subtype

