ds_arr=("emotions_new.mat" "yeast_new.mat" "pascal_new.mat" "pca_mir5k.mat" "esp_test.mat")
r_arr=(1 2 3)
p_arr=(3 7)

for ds in ${dataset_array[@]}
do
    for r in ${r_array[@]}
    do
        for p in ${p_array[@]}
        do
            python -u ${exp}.py --dataset ${ds} --r ${r} --p ${p}
        done
    done
done