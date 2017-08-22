
function change_train_conf()
{
    f=$1
    sed -i "s/^data *=.*/data = learn_LTR\/sample_data\/Fold${f}\/lightgbm_train.txt/" train.conf
    sed -i "s/^valid_data *=.*/valid_data = learn_LTR\/sample_data\/Fold${f}\/lightgbm_vali.txt/" train.conf
    sed -i "s/^output_model *=.*/output_model = model\/lgbm${f}/" train.conf
}
function change_predict_conf()
{
    f=$1
    iter=$2
    sed -i "s/^data *=.*/data = learn_LTR\/sample_data\/Fold${f}\/lightgbm_test.txt/" predict.conf
    sed -i "s/^input_model *=.*/input_model =model\/lgbm${f}/" predict.conf
    sed -i "s/^output_result *=.*/output_result =score\/lgbm\/lgbm.${f}.sc/" predict.conf
    sed -i "s/^num_iteration_predict *=.*/num_iteration_predict=${iter}/" predict.conf
}

mkdir -p score/lgbm
mkdir -p report

for i in 1 2 3 4 5;do
    change_train_conf $i
    lightgbm config=train.conf | tee logg
    best=$(python choose_best.py logg)
    echo best Iteration = $best
    # 发现无论验证集取最好的模型(即树个数), 效果还不如直接取默认100
    change_predict_conf $i 100 #$best
    lightgbm config=predict.conf
done

python gen_report.py
cat report/report_mean.csv

#for j in `seq 1 100`;do
#    for i in 1 2 3 4 5;do
#        change_predict_conf $i $j
#        lightgbm config=predict.conf >/dev/null
#    done
#    python gen_report.py
#    echo $j
#    cat report/report_mean.csv
#    echo "-------------------------"
#done
