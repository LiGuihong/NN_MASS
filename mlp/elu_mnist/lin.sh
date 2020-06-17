

for num_segs in 20
do
    for depths in 20
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=60 --tc=$tc
        done
    done
done



for num_segs in 20
do
    for depths in 28
    do
        for tc in 0 1 2 3 4 5 6 7 8 9
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=60 --tc=$tc
        done
    done
done
