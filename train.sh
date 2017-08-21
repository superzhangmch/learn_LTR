function run_tr()
{
    model=$1
    CUDA_VISIBLE_DEVICES=3  python ltr_algorithm.py $model 1
    CUDA_VISIBLE_DEVICES=3  python ltr_algorithm.py $model 2
    CUDA_VISIBLE_DEVICES=3  python ltr_algorithm.py $model 3
    CUDA_VISIBLE_DEVICES=3  python ltr_algorithm.py $model 4
    CUDA_VISIBLE_DEVICES=3  python ltr_algorithm.py $model 5
}
for model in fea_0 fea_1 fea_2 fea_3 fea_4 fea_5 fea_6 fea_7 fea_8 fea_9 fea_10 fea_11 fea_12 fea_13 fea_14 fea_15 fea_16 fea_17 fea_18 fea_19 fea_20 fea_21 fea_22 fea_23 fea_24 fea_25 fea_26 fea_27 fea_28 fea_29 fea_30 fea_31 fea_32 fea_33 fea_34 fea_35 fea_36 fea_37 fea_38 fea_39 fea_40 fea_41 fea_42 fea_43 fea_44 fea_45 lambdarank ranknet ranknet_speedup ranking_svm log_loss_binary_classify mse_regression lambdarank_slow; do
   echo $model
   mkdir -p score/$model/
   run_tr $model
done
