from dicom.rt_utils.rt_utils_mod import RTStructBuilder
import numpy as np
import os

# Merge an RTSTRUCT and create grouped ROIs based on predefined organ groups
def merge_rtstruct(rtstruct_path, ct_folder, output_path, groups):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rt_src = RTStructBuilder.create_from(
        dicom_series_path=ct_folder,
        rt_struct_path=rtstruct_path
    )
    rt_dst = RTStructBuilder.create_new(dicom_series_path=ct_folder)

    # Copy original ROIs
    for name in rt_src.get_roi_names():
        try:
            m = rt_src.get_roi_mask_by_name(name)
            if m is not None:
                rt_dst.add_roi(mask=m.astype(bool), name=name)
        except Exception:
            pass  # skip invalid ROIs

    # Case-insensitive ROI lookup
    def get_mask_by_name_ci(builder, name):
        if name in builder.get_roi_names():
            return builder.get_roi_mask_by_name(name)
        lname = name.strip().lower()
        for s in builder.get_roi_names():
            if s.strip().lower() == lname:
                return builder.get_roi_mask_by_name(s)
        return None

    # Build grouped ROIs
    # Build grouped ROIs
    for group_name, recipes in groups.items():

        for recipe in recipes:

            masks = []

            for roi in recipe:

                m = get_mask_by_name_ci(rt_dst, roi)

                if m is None:
                    m = get_mask_by_name_ci(rt_src, roi)

                # This recipe fails if one ROI is missing
                if m is None:
                    masks = []
                    break

                masks.append(m.astype(bool))

            # Successful recipe
            if masks:

                merged = np.any(np.stack(masks, axis=0), axis=0)

                if np.any(merged):
                    rt_dst.add_roi(mask=merged, name=group_name)
                    print(f"[RTSTRUCT] {group_name}: built from {recipe}")

                break

    rt_dst.save(output_path)
    print("[RTSTRUCT] Grouped saved:", output_path)

def merge_rtstruct_files(CT, RTSTRUCT_LOCATION, RTSTRUCTS):
    import os
    from dicom.rt_utils.rt_utils_mod import RTStructMerger

    # Merge the first two RTSTRUCTs
    merged_rt_struct = RTStructMerger.merge_rtstructs(
        dicom_series_path=CT,
        rt_struct_path1=RTSTRUCTS[0],
        rt_struct_path2=RTSTRUCTS[1],
    )
    merged_rt_struct.save(os.path.join(RTSTRUCT_LOCATION, "segmentations.dcm"))

    RTSTRUCT = os.path.join(RTSTRUCT_LOCATION, "segmentations.dcm")
    merged_path = RTSTRUCTS[0]

    # Iteratively merge remaining RTSTRUCTs (if any)
    for path in RTSTRUCTS[1:]:
        merged = RTStructMerger.merge_rtstructs(
            dicom_series_path=CT,
            rt_struct_path1=merged_path,
            rt_struct_path2=path,
        )
        merged.save(RTSTRUCT)
        merged_path = RTSTRUCT

    print("[RTSTRUCT] Merged saved:", RTSTRUCT)
    return RTSTRUCT
