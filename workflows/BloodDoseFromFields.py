from simulation import ExpandFlowModel, TemporalDistribution, DoseRate, CompartmentDose, Patient
from PlotDoseDistribution import plot_dose_distribution
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def blood_dose_distribution(
    simulation_params,
    patient_params,
    treatment_params,
    patient_dir,
    blood_path,
    plan_name,
    run_number,
):
    patient = Patient()
    patient.read_from_numpy(patient_dir, patient_params.organs, plot=True)
    patient_params.organs = list(patient.seg_organs.keys())

    dvh_dir = os.path.join(patient_dir, "DVHs")
    patient.write_dvh(dvh_dir, patient_params.organs, plan_name)
    # patient.get_tumor_volume_fraction(patient_params['tumor_site'], 'tumor')
    # patient_params['tumor_volume_fraction'] = patient.tumor_volume_fraction
    print('Tumor volume fraction: {:.3f}'.format(patient_params.tumor_volume_fraction))

    # ======= Step 1. Initialize simulation ============================= #

    filename = ROOT / "input" / "phantom" / "ICRP89_compartment_model.xlsx"
    model = ExpandFlowModel(filename, patient_params, simulation_params)
    # add the tumor:
    if 'tumor' in patient_params.organs:
        model.split_box_parallel('tumor', box_dict=patient_params)
    # ============================================================== #

    # ======== Step 2. Generate distribution ======================= #
    blood = TemporalDistribution(model)
    if simulation_params['generate_new']:
        model.construct_weibull()
        blood.generate_from_weibull()
        blood.save(blood_path)
    else:
        blood.load(blood_path)

    # ======== Plot stuff for verification ========================= #
    # blood.plot_time_distributions(['lung'])
    # blood.plot_inflow_outflow(['lung'])
    # blood.plot_volumes_over_time()
    # blood.plot_normalized_volume_over_time(['lung', 'tumor'])
    # blood.plot_final_blood_volumes()
    # ============================================================== #

    # ======== Step 3. Accumulate dose ============================= #
    dose = DoseRate(patient, n_fractions=treatment_params['nr_fractions'],
                    total_beam_on_time=treatment_params['total_beam_on_time'])
    dose.get_dose_rate()

    compartment_ids = [[i for i, name in enumerate(model.names) if organ in name] for organ in patient_params.organs]
    dose_contributions = {}
    for organ, compartment_id in zip(patient_params.organs, compartment_ids):
        blood_dose = CompartmentDose(blood.path, model.dt)
        if simulation_params['random_walk']:

            # Skip structures with no voxels
            if organ not in patient.seg_organs:
                print(f"[HEDOS] Skipping {organ}: no segmentation")
                continue

            if np.sum(patient.seg_organs[organ]) == 0:
                print(f"[HEDOS] Skipping {organ}: empty mask")
                continue

            # Skip structures with no blood compartment
            if len(compartment_id) == 0:
                print(f"[HEDOS] Skipping {organ}: no compartment")
                continue

            try:
                blood_dose.prepare(
                    patient.gridpoints,
                    patient.seg_organs[organ],
                    down_sample=(2, 2, 1)
                )
            except ValueError:
                print(f"[HEDOS] Skipping {organ}: random walk preparation failed")
                continue

            for start_time, beam_on_time in zip(
                    treatment_params['start_times'],
                    treatment_params['beam_on_times']
            ):
                try:
                    blood_dose.add_dose_random_walk(
                        dose.dose_rate_func,
                        compartment_id,
                        start_time=start_time,
                        beam_on_time=beam_on_time
                    )
                except ValueError:
                    print(f"[HEDOS] Skipping {organ}: no valid random walk voxels")
                    break
        else:
            dose_rate_hist = dose.get_dose_rate_hist(organ)
            for start_time, beam_on_time in zip(treatment_params['start_times'], treatment_params['beam_on_times']):
                blood_dose.add_dose(dose_rate_hist, compartment_id, start_time=start_time, beam_on_time=beam_on_time)
        dose_contributions[organ] = blood_dose.dose

    blood_dose_total = CompartmentDose(blood.path, model.dt)
    blood_dose_total.dose = sum(list(dose_contributions.values()))
    if simulation_params['accumulate']:
        blood_dose_total.repeat(treatment_params['nr_fractions'])

    blood_volumes = [sum(list(model.volumes[ids])) for ids in compartment_ids]
    dose.calculate_mean_blood_dose(model.total_volume, blood_volumes, accumulate=simulation_params['accumulate'])
    plot_dose_distribution(blood_dose_total,dose_contributions,plan_name,run_number,dose.MBD,)

    return blood_dose_total.dose
    # ============================================================== #
