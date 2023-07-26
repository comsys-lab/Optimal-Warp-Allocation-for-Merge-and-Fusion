nvcc gms_main.cu -o gms

if [ $# -ne 1 ]
then
    echo "실행 방법:"
    echo "./gms.sh thread"
    echo "./gms.sh block"
elif [ $1 = "thread" ]
then
    echo "구현 중"
    # echo $1
    # > gms_results_thread.txt
    # ncu --metrics gpu__time_duration.sum -k fused_two_conv_thread gms 1 | grep gpu__time_duration.sum >> gms_results_thread.txt
    # echo "Test_results_inter_thread.txt 생성"
elif [ $1 = "block" ] 
then
    echo $1
    > gms_results_block.txt
    ncu --metrics gpu__time_duration.sum -k fused_two_conv_block gms 0 | grep gpu__time_duration.sum >> gms_results_block.txt
    # ncu --metrics gpu__time_duration.sum -k gms_conv gms 0 | grep gpu__time_duration.sum >> gms_results_block.txt
    echo "Test_results_inter_block.txt 생성"
else
    echo "실행 방법:"
    echo "./gms.sh thread"
    echo "./gms.sh block"
fi