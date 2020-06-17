for num_segs in 20
do
    for depths in 16 24 32
    do
        for tc in 1
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=60 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 20 28 32
    do
        for tc in 3
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=60 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 16 20 24
    do
        for tc in 8
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=60 --tc=$tc
        done
    done
done