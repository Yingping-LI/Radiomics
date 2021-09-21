#!/usr/bin/env python
# coding: utf-8


#========================================
# - "results_bathpath":  -TCGA: "G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results/ArrangeResults",
#                        -BraTs2021: "G://PhDProjects/RadiogenomicsProjects/BraTS2021/Results/ArrangeResults"
# - "imagetype_dict": image type chosen for extracting features.

'''
Basic Settings for the code
'''
global_basic_settings={
    "results_bathpath": "G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results/TCGA_MGMT_site_lbp-3D-m1",
    #"results_bathpath": "G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results/TCGA_IDH_site_exponential",
    #"results_bathpath": "G://PhDProjects/RadiogenomicsProjects/BraTS2021/Results/ArrangeResults",
    #"G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results/arrange_results_wavelet" ,
    #"G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results/ArrangeResults",
    "imagetype_dict": {"TCGA_IDH": "exponential",
                       "TCGA_MGMT": "lbp-3D-m1",#"LBP-3d-m1" or "LBP-3d-m1"
                       "BraTS2021": "lbp-3D-m1",
                      }
}

def get_basic_settings():   

    return global_basic_settings






