

for num_segs in 20
do
    for depths in 20
    do
        for tc in 1
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=100 --tc=$tc
        done
    done
done

for num_segs in 20
do
    for depths in 20
    do
        for tc in 7
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=100 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 32
    do
        for tc in 1
        do

            python main_elu_mnist.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=100 --tc=$tc
        done
    done
done

