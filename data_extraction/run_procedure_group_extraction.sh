python3 extract_procedure_groupings.py --include='initial hospital care' --exclude='' --group_name=hospitalization --group_name_readable='Hospitalization'
python3 extract_procedure_groupings.py --include='surgical%surgery' --exclude='' --group_name=surgery --group_name_readable='Surgery'
python3 extract_procedure_groupings.py --include='nursing' --exclude='subsequent%therapy%collection%ventilation%discharge%annual' --group_name=nursing --group_name_readable='Nursing'
python3 extract_procedure_groupings.py --include='emergency department visit' --exclude='' --group_name=emergency_visit --group_name_readable='Emergency visit'
python3 extract_procedure_groupings.py --include='outpatient%office' --exclude='inpatient%discharge%cardiac%glucose%dialysis%chemotherapy%intranasal' --group_name=office_visit --group_name_readable='Office visit'
python3 extract_procedure_groupings.py --include='collection of venous blood' --exclude='' --group_name=blood_test --group_name_readable='Blood test'
python3 extract_procedure_groupings.py --include='electrocardiogram' --exclude='rhythm' --group_name=electrocardiogram --group_name_readable='Electrocardiogram'
python3 extract_procedure_groupings.py --include='eye%ophthalmic%visual field%lens%retina%ophthalmological%ocular%refractive%visual acuity' --exclude='eyelid%retinacular%retinaculum' --group_name=eye_services --group_name_readable='Eye services'
python3 extract_procedure_groupings.py --include='immunization%vaccine' --exclude='' --group_name=vaccination --group_name_readable='Vaccination'
python3 extract_procedure_groupings.py --include='echocardiography' --exclude='' --group_name=echocardiography --group_name_readable='Echocardiography'
python3 extract_procedure_groupings.py --include='debridement of nail' --exclude='' --group_name=nail_debridement --group_name_readable='Nail debridement'
python3 extract_procedure_groupings.py --include='computed tomography' --exclude='fluoroscopy' --group_name=ct_scan --group_name_readable='CT scan'
python3 extract_procedure_groupings.py --include='physical therapy%therapeutic procedure%manual therapy%therapeutic activities%chiropractic%application of a modality' --exclude='radiotherapeutic' --group_name=physical_therapy --group_name_readable='Physical therapy'
python3 extract_procedure_groupings.py --include='arthrocentesis' --exclude='' --group_name=arthrocentesis --group_name_readable='Arthrocentesis'
python3 extract_procedure_groupings.py --include='radiologic examination, chest' --exclude='' --group_name=chest_xray --group_name_readable='Chest X-ray'
python3 extract_procedure_groupings.py --include='subsequent hospital care' --exclude='' --group_name=subsequent_hospital_care --group_name_readable='Subsequent hospital care'
python3 extract_procedure_groupings.py --include='bladder capacity' --exclude='' --group_name=bladder_measurement --group_name_readable='Bladder measurement'
python3 extract_procedure_groupings.py --include='injection' --exclude='' --group_name=injection --group_name_readable='Injection'
python3 extract_procedure_groupings.py --include='screening mammography%screening digital breast tomosynthesis' --exclude='' --group_name=breast_cancer_screening --group_name_readable='Breast cancer screening'
python3 extract_procedure_groupings.py --include='lesion' --exclude='lesion detection' --group_name=lesion_proc --group_name_readable='Lesion-related procedures'
python3 extract_procedure_groupings.py --include='critical care' --exclude='telehealth' --group_name=critical_care --group_name_readable='Critical care'
python3 extract_procedure_groupings.py --include='device evaluation' --exclude='' --group_name=cardiac_device_monitoring --group_name_readable='Cardiac device monitoring'
python3 extract_procedure_groupings.py --include='bone density study' --exclude='' --group_name=bone_density_study --group_name_readable='Bone density study'
python3 extract_procedure_groupings.py --include='myocardial perfusion imaging' --exclude='' --group_name=myocardial_perfusion_imaging --group_name_readable='Myocardial perfusion imaging'
python3 extract_procedure_groupings.py --include='duplex scan' --exclude='' --group_name=vascular_duplex_scan --group_name_readable='Vascular duplex scan'
python3 extract_procedure_groupings.py --include='inpatient consultation' --exclude='' --group_name=inpatient_consultation --group_name_readable='Inpatient consultation'
python3 extract_procedure_groupings.py --include='impacted cerumen' --exclude='' --group_name=earwax_removal --group_name_readable='Earwax removal'
python3 extract_procedure_groupings.py --include='radiologic examination' --exclude='' --group_name=radiologic_exam --group_name_readable='Radiologic exam'
python3 extract_procedure_groupings.py --include='audiometry%hearing%pure tone' --exclude='' --group_name=hearing_exam --group_name_readable='Hearing exam'
python3 extract_procedure_groupings.py --include='blood typing' --exclude='' --group_name=blood_typing --group_name_readable='Blood typing'
python3 extract_procedure_groupings.py --include='self-care%home management' --exclude='' --group_name=self_care_training --group_name_readable='Self-care training'
python3 extract_procedure_groupings.py --include='intravenous infusion' --exclude='inhaled' --group_name=intravenous_infusion --group_name_readable='Intravenous infusion'
python3 extract_procedure_groupings.py --include='preventive medicine' --exclude='' --group_name=preventive_medicine_evaluation --group_name_readable='Preventive medicine evaluation'
python3 extract_procedure_groupings.py --include='surgical pathology' --exclude='' --group_name=surgical_pathology --group_name_readable='Surgical pathology'
python3 extract_procedure_groupings.py --include='ultrasound' --exclude='guidance%guided%tomography%ultrasound, when performed%non-imaging' --group_name=ultrasound --group_name_readable='Ultrasound'
python3 extract_procedure_groupings.py --include='cytopathology, cervical or vaginal' --exclude='' --group_name=cervical_screening --group_name_readable='Cervical screening'
python3 extract_procedure_groupings.py --include='gynecolog' --exclude='' --group_name='gynecology' --group_name_readable='Gynecology'
python3 extract_procedure_groupings.py --include='spirometry' --exclude='' --group_name='spirometry' --group_name_readable='Spirometry'
python3 extract_procedure_groupings.py --include='esophagogastroduodenoscopy' --exclude='' --group_name='esophagogastroduodenoscopy' --group_name_readable='Esophagogastroduodenoscopy'
python3 extract_procedure_groupings.py --include='colonoscopy' --exclude='' --group_name='colonoscopy' --group_name_readable='Colonoscopy'
python3 extract_procedure_groupings.py --include='oximetry' --exclude='' --group_name='oximetry' --group_name_readable='Oximetry'
python3 extract_procedure_groupings.py --include='antibody screen' --exclude='' --group_name='antibody_screen' --group_name_readable='Antibody screen'