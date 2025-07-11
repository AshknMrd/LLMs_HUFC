docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task002_nodule_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task005_recist_timeline_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task007_nodule_diameter_presence_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task010_prostate_radiology_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task011_prostate_pathology_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task016_recist_lesion_size_presence_clf.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task019_prostate_volume_reg.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task020_psa_reg.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task021_psad_reg.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/chestct-lung-nodules:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task023_nodule_diameter_reg.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/rumc-jbz/recist:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task024_recist_lesion_size_reg.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/Task025_anonymisation_ner.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated/algorithm-input:/input/federated/algorithm-input \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/algorithm-input:/input/algorithm-input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated/debug-input:/output/federated/debug-input \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated/debug-test-set:/output/federated/debug-test-set \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/make_debug_splits.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated/algorithm-input:/input/federated/algorithm-input \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/algorithm-input:/input/algorithm-input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated/test-set:/input/federated/test-set \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/test-set:/input/test-set:ro \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/make_test_splits.py

docker run --rm \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed:/input:ro \
    -v /Users/joeranbosma/repos/dragon_data/preprocessed/federated:/output \
    dragon_prep:latest python /opt/app/dragon_prep/src/dragon_prep/federated/make_synthetic_splits.py
