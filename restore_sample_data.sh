function make_data()
{
  fold=$1
  s1=$2
  s2=$3
  s3=$4
  s4=$5
  s5=$6
  mkdir -p Fold${fold}
  cat S$2.txt S$3.txt S$4.txt > Fold${fold}/train.txt
  cp S$5.txt Fold${fold}/vali.txt
  cp S$6.txt Fold${fold}/test.txt
}

cd sample_data
tar -zxvf data_set.tar.gz
make_data 1 1 2 3 4 5
make_data 2 2 3 4 5 1
make_data 3 3 4 5 1 2
make_data 4 4 5 1 2 3
make_data 5 5 1 2 3 4
cd -
#Fold    Training.txt    Validation.txt Test.txt
#Fold1   S1 S2  S3      S4              S5
#Fold2   S2 S3  S4      S5              S1
#Fold3   S3 S4  S5      S1              S2
#Fold4   S4 S5  S1      S2              S3
#Fold5   S5 S1  S2      S3              S4

