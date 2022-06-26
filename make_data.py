from tractogram_data import Subject
import nibabel.streamlines as nibs
import nibabel as nib
import pandas as pd
from tqdm import tqdm
subject_number = 2

def load_tract(subject_number:int, tract:str, method:str, side:str, clean=False,flipped=False,ACT=False):
    """Flexibly loads tracts based on input features (string)"""
     
    # check if clean and or flipped: will influence template_path
    if clean:
        if flipped:
            # clean and flipped
            template_path = f"./data/Cleaner_TCKs/QBX_filtered_flipped/sub-PT{subject_number:03d}_QBXf_TCKs_flip/"
            template_vol_path = f"./data/Cleaner_TCKs/QBX_filtered_flipped/sub-PT{subject_number:03d}_QBXf_TCK_maps_flip/"
            #AF_wprecentral_LT_fin_map_ACT_iFOD2_inNat_flip.nii.gz
            
        else:
            # clean and NOT flipped
            template_path = f"./data/Cleaner_TCKs/QBX_filtered/sub-PT{subject_number:03d}_QBXf_TCKs/" 
            template_vol_path = f"./data/Cleaner_TCKs/QBX_filtered/sub-PT{subject_number:03d}_QBXf_TCK_maps/"
            #AF_wprecentral_LT_fin_map_ACT_iFOD2_inNat.nii.gz
    else:
           
        if flipped:
            # NOT clean and flipped
            template_path = f"./data/S61759_BIDS_fMRI/BIDS/derivatives/Warping_2_native/flipped/sub-PT{subject_number:03d}_TCKs_flip/" 
            template_vol_path = f"./data/S61759_BIDS_fMRI/BIDS/derivatives/Warping_2_native/flipped/sub-PT{subject_number:03d}_TCKs_flip/" 
            #AF_wprecentral_LT_all_fin_BT_map_ACT_iFOD2_inNat_flip.nii.gz

        else:
            # NOT clean and NOT flipped
            template_path = f"./data/S61759_BIDS_fMRI/BIDS/derivatives/Warping_2_native/TCKs_Fin_full_bkup/sub-PT{subject_number:03d}_TCKs_warping/TCKs/"
            template_vol_path= f"./data/S61759_BIDS_fMRI/BIDS/derivatives/Warping_2_native/TCKs_Fin_full_bkup/sub-PT{subject_number:03d}_TCKs_warping/TCK_maps/"
            #AF_wprecentral_LT_all_fin_BT_map_ACT_iFOD2_inNat.nii.gz

    #check for special strings: ACT and flip
    if ACT:
        act_string = "ACT_"
    else:
        act_string = ""
    
    if flipped:
        flip_string = "_flip"
    else:
        flip_string= ""
    
    #construct template file string
    template_file = f"{tract}_{side}T_fin_BT_{act_string}{method}_inNat{flip_string}.tck"

    if clean:
        template_vol_file = f"{tract}_{side}T_fin_map_{act_string}{method}_inNat{flip_string}.nii.gz"
    else:
         template_vol_file = f"{tract}_{side}T_all_fin_BT_map_{act_string}{method}_inNat{flip_string}.nii.gz"

    #load the .tck file with nibabel.streamlines (as nibs here) and return
    file_path = template_path + template_file
    vol_path = template_vol_path + template_vol_file

    tract_file = nibs.load(file_path)
    vol_img = nib.load(vol_path)

    return tract_file, vol_img, file_path

def load_data():
    df2 = pd.DataFrame()
    for subject_number in range(100):
        for method_logic in ["iFOD2","Tensor_Prob"]:
            for flipped_logic in [True, False]:
                for clean_logic in [True, False]:
                    for ACT_logic in [True, False]:
                        try:
                            if flipped_logic==False:
                                #if not flipped, load normally
                                tract_L, vol_img, _ = load_tract(subject_number=subject_number,tract ="IFOF",method=method_logic,side="L",
                                                        flipped=flipped_logic,clean=clean_logic,ACT=ACT_logic)
                                tract_R, _, _ = load_tract(subject_number=subject_number,tract ="IFOF",method=method_logic,side="R",
                                                        flipped=flipped_logic,clean=clean_logic,ACT=ACT_logic)
                            else:
                                #if flipped load the opposite files to L and R
                                tract_L, vol_img, _  = load_tract(subject_number=subject_number,tract ="IFOF",method=method_logic,side="R",
                                                        flipped=flipped_logic,clean=clean_logic,ACT=ACT_logic)
                                tract_R, _, _ = load_tract(subject_number=subject_number,tract ="IFOF",method=method_logic,side="L",
                                                        flipped=flipped_logic,clean=clean_logic,ACT=ACT_logic)

                        
                            sub_meta = {"subject_id":subject_number,
                                        "method":method_logic,
                                        "tract":"AF",
                                        "flipped":flipped_logic,
                                        "clean":clean_logic,
                                        "ACT":ACT_logic}
                            print(sub_meta)

                            sub = Subject(tract_L, tract_R, vol_img, sub_meta)
                            df = sub.df
                            df2 = pd.concat([df2, df])

                        except:
                            sub_meta = {"subject_id":subject_number,
                                        "method":method_logic,
                                        "tract":"AF",
                                        "flipped":flipped_logic,
                                        "clean":clean_logic,
                                        "ACT":ACT_logic}
                            print("")
                            print(f"NOT SUCCESFUL FOR: {sub_meta}")
                            print("")
                            continue

        df2.to_csv("tractography_data.csv")

def combine_tract_fmri():
    df = pd.read_csv("tractography_data.csv")
    print(df.head())

    df_fmri = pd.read_csv("advanced_tract_fmri_combined.csv")
    print(df_fmri.head())

    df_fmri["subject_id"] = df_fmri.patient

    df_merge = df_fmri[['LI_fmri','subject_id']]
    df_test = pd.merge(df, df_merge,on="subject_id")

    n_sub_right = len(set(df_test.subject_id[df_test.LI_fmri < 0]))
    n_sub_left = len(set(df_test.subject_id[df_test.LI_fmri > 0]))
    n_subs = len(set(df_test.subject_id))

    print(f"""
    Number left right lateralized: {n_sub_right}
    Number left lateralized: {n_sub_left}
    Total number of subs: {n_subs}
    """)

    #make fMRI values negative for all the "flipped" patients

    df_test["LI_fmri_flipped"] = -df_test.LI_fmri
    print(df_test.info())
    df_test.drop(columns = ["Unnamed: 0"])
    print(df_test.info())

    df_test.to_csv("tract_fmri_data.csv")

if __name__ == '__main__':
    # L, V, _ = load_tract(1,"AF","iFOD2","L",True, True, True)
    # R, V, _ = load_tract(1,"AF","iFOD2","R",True, True, True)

    # sub = Subject(L,R,V)
    load_data()
    combine_tract_fmri()