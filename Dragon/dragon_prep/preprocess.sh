docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task101_Example_sl_bin_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task102_Example_sl_mc_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task103_Example_mednli.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task104_Example_ml_bin_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task105_Example_ml_mc_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task106_Example_sl_reg.py
docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task107_Example_ml_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task108_Example_sl_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task109_Example_ml_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rijnstate/cinemri:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task001_adhesion_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task002_nodule_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/kidney:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task003_kidney_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/bcc:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task004_skin_case_selection_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task005_recist_timeline_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pathology-lung:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task006_lung_tumor_origin_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task007_nodule_diameter_presence_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pancreas:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task008_pdac_size_presence_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pancreas:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task009_pdac_diagnosis_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task010_prostate_radiology_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task011_prostate_pathology_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pathology-lung:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task012_pathology_tissue_type_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pathology-lung:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task013_pathology_tissue_origin_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/mednli:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task014_textual_entailment_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/colon:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task015_colon_pathology_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task016_recist_lesion_size_presence_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pancreas:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task017_pdac_attributes_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/hip:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task018_osteoarthritis_clf.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task019_prostate_volume_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task020_psa_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task021_psad_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/pancreas:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task022_pdac_size_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task023_nodule_diameter_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task024_recist_lesion_size_reg.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task025_anonymisation_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/terminology:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task026_medical_terminology_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/prostate-biopsy:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task027_prostate_biopsy_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/rumc/bcc:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/Task028_skin_pathology_ner.py

docker run --rm -it \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/algorithm-input:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/debug-input:/output/debug-input \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/debug-test-set:/output/debug-test-set \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/make_debug_splits.py
