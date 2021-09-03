#!/usr/bin/env python
# coding: utf-8

import pydicom  

"""
Class: extract the scanner info from an .dcm image.
"""
class DcmData(object):
    def __init__(self, dcm_imagepath):
        self.dcm_imagepath=dcm_imagepath
        self.dcm_info=pydicom.read_file(self.dcm_imagepath)
        
        interested_attribute_list=['PatientID', 'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'SpacingBetweenSlices', 
                                   'ReconstructionDiameter', 'AcquisitionMatrix', 'PixelSpacing', 'SeriesDescription']
        
        self.Infos={}
        ## Find the dcm info for the attributes in interested_attribute_list.
        for interested_attribute in interested_attribute_list:
            self.Infos[interested_attribute]=None
            
            # find the value for each interested attribute.
            for attribute in self.dcm_info:      
                if attribute.keyword==interested_attribute:
                    self.Infos[interested_attribute]=attribute.value
                    
        ## Normalize the magnetic field strength, get the image modality
        self.normalize_magnetic_field_strength()            
        self.normalize_SeriesDescription()
                    
    def get_Infos(self):
        return self.Infos
    
   
    def normalize_magnetic_field_strength(self):         
        #Approximate to 1.5T or 3T
        self.Infos["MagneticFieldStrength_normalized"]=self.Infos["MagneticFieldStrength"]
        if self.Infos["MagneticFieldStrength"] is not None:
            if self.Infos["MagneticFieldStrength"]==30000 or abs(self.Infos["MagneticFieldStrength"]-3)<0.5:
                self.Infos["MagneticFieldStrength_normalized"]= 3
            elif self.Infos["MagneticFieldStrength"]==15000  or abs(self.Infos["MagneticFieldStrength"]-1.5)<0.5:
                self.Infos["MagneticFieldStrength_normalized"]= 1.5
        
        # is 3T or not?
        self.Infos["is_3T"]= (self.Infos["MagneticFieldStrength_normalized"]==3) + 0
    
    
    def normalize_SeriesDescription(self):
        if self.Infos["SeriesDescription"] is None:
            self.Infos["SeriesDescription_normalized"]="None"
            
        else:
            SeriesDescription=self.Infos["SeriesDescription"].lower()
            if "flair" in SeriesDescription:
                self.Infos["SeriesDescription_normalized"]="flair"

            elif "t2" in SeriesDescription:
                self.Infos["SeriesDescription_normalized"]="t2"

            elif ("post" in SeriesDescription) or ("+" in SeriesDescription) or ("t1wce" in SeriesDescription):
                self.Infos["SeriesDescription_normalized"]="t1ce"

            else:   
                #SeriesDescription.contains("t1") or SeriesDescription.contains("fspgr 3d"):
                self.Infos["SeriesDescription_normalized"]="t1"

        
    def show_all_attributes(self):
        for attribute in self.dcm_info:
            if attribute.keyword != '' and attribute.keyword != 'PixelData':
                print('--{}:  {}'.format(attribute.keyword, attribute.value))
                
