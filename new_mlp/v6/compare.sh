
for num_segs in 20
do
    for depths in 4 5 6 8 10 12 16 20
    do
        for tc in 0
        do

            python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
            python main_random.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc

        done
    done
done