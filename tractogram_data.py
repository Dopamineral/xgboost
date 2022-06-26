import numpy as np
import nibabel.streamlines as nibs
import nibabel as nib
from numpy.linalg import norm
from skimage.measure import marching_cubes, mesh_surface_area
from dipy.io.streamline import load_tck
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map
import pandas as pd

class Subject():
    """
    Subject class to hold all the tract info / metrics / etc...
    """

    def __init__(self, tract_L, tract_R, vol_img, meta_data:dict = {}): 
        
        # you need this reference nii file to succesfully run the streamlines2vol script
        self.vol_img = vol_img
        #Store subject metadata
        self.meta_data = meta_data

        #load tract before calling the class
        self.tract_L = tract_L
        self.volume_L = self.streamline2volume(self.tract_L)
        self.metrics_L = self.calculate_metrics(self.tract_L, self.volume_L)

        #Right sided metrics
        self.tract_R = tract_R
        self.volume_R = self.streamline2volume(self.tract_R)
        self.metrics_R = self.calculate_metrics(self.tract_R, self.volume_R)

        #Combine into one df

        self.df = self.make_df()

    def streamline2volume(self,tract):
            """
            Ceate volume of tract starting from the .tcks file

            Input:
                - streamlines data

            Returns:
                - 3D volume (x,y,z) of the subject tract
            """
        
            reference = self.vol_img
            ref_affine= reference.affine
            ref_dim = reference.header["dim"][1:4]
            streamlines = tract.tractogram.streamlines 
            
            tract_vol = density_map(streamlines, affine=ref_affine,vol_dims=ref_dim)

            output_vol =  np.where(tract_vol>0.5, 1, 0) # binarize the volume

            return output_vol 

    def calculate_metrics(self,tract, tract_vol):
        """
        Calculates all the metrics from the papers cited.

        Input:
            - streamlines data
            - 3D volume data

        Returns:
            - dictionary of metrics for the subject
        """
        
        # Functions to calculate the metrics 
        def stream_length(stream):
            '''
            calculates lenght of a stream as part of calculating tract length in:
            https://www.sciencedirect.com/science/article/pii/S1053811920308156?via%3Dihub
            table 1
            '''
            stream_sum = 0
            for tt in range(stream.shape[0]-1):
                stream_sum += norm(stream[tt]-stream[tt+1])
            return stream_sum


        def tract_length(tract):
            '''
            calculates tract lenght as defined in:
            https://www.sciencedirect.com/science/article/pii/S1053811920308156?via%3Dihub
            table 1
            '''
            tract_streams = tract.streamlines
            stream_sums = []
            for stream in tract_streams:
                stream_sums.append(stream_length(stream))
                
            output = np.mean(stream_sums)
            return output

        def tract_span(tract):
            '''
            calculates tract span as defined in:
            https://www.sciencedirect.com/science/article/pii/S1053811920308156?via%3Dihub
            table 1
            '''
            tract_streams = tract.streamlines
            span_sums = []
            for stream in tract_streams:
                span_sums.append(norm(stream[0] - stream[-1]))
            tract_span = np.mean(span_sums)

            return tract_span


        def tract_diameter(tract, tract_vol):
            length_metric = tract_length(tract)
            N_voxels = np.sum(tract_vol)
            volume = N_voxels
            diameter = 2*np.sqrt(volume / (np.pi*length_metric))
            return diameter


        def tract_surface_area(tract_vol):

            verts, faces, _, _ = marching_cubes(tract_vol)
            surface_area = mesh_surface_area(verts, faces)
            return surface_area

        # Calculate all the metrics and assign to variables
        length_metric = tract_length(tract)
        span_metric = tract_span(tract)
        curl_metric = length_metric / span_metric
        diameter_metric = tract_diameter(tract, tract_vol)
        elongation_metric = length_metric / diameter_metric
        volume_metric = np.sum(tract_vol)
        surface_area_metric = tract_surface_area(tract_vol)
        irregularity_metric = surface_area_metric / (np.pi*diameter_metric*length_metric)


        # Construct final dictionary to output
        metrics_dict = {'tract_length':[length_metric], #putting in brackets to later import in dataframe
                        'tract_span': [span_metric],
                        'tract_curl':[curl_metric],
                        'tract_diameter':[diameter_metric],
                        'tract_elongation':[elongation_metric],
                        'tract_volume':[volume_metric],
                        'tract_surface_area':[surface_area_metric],
                        'tract_irregularity':[irregularity_metric]
                        }



        return metrics_dict


    def make_df(self):

        def update_keys_append(d:dict, suffix:str):
            "Adds a particular suffix to dictionary keys"
            old_keys = list(d.keys())

            for ii in range(len(old_keys)):
                old_key = old_keys[ii]
                new_key = old_key+suffix

                d[new_key] = d[old_key]
                d.pop(old_key)

            return d

        update_keys_append(self.metrics_L,'_L')
        update_keys_append(self.metrics_R,'_R')
        
        # putting ** before dictionary unpacks it

        combined_dict = {**self.meta_data,**self.metrics_L,**self.metrics_R}

        df = pd.DataFrame.from_dict(combined_dict)
        return df

if __name__ == "__main__":


    sub_path_L = "./sub-001/TCKs/AF_wprecentral_LT_fin_BT_ACT_iFOD2_inNat.tck"
    sub_path_R = "./sub-001/TCKs/AF_wprecentral_RT_fin_BT_ACT_iFOD2_inNat.tck"


    sub_metadata = {"id":1,
                    "sex":None,
                    "age":None,
                    "tract_name":None}

    sub = Subject(sub_path_L,
                sub_path_R,
                sub_metadata)

    print(sub.df.T)

    import streamlit
    print(streamlit.__version__)

