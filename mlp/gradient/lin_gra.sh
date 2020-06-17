


for num_segs in 20
do
    for depths in 16
    do
        for tc in 0 20 40 60 80 100 120 140
        do

            python new_test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 20
    do
        for tc in 0 20 40 60 80 100 120 140 160 180 200
        do

            python new_test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 24
    do
        for tc in 0 20 40 60 80 100 120 140 160 180 200 220 240 260
        do

            python new_test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 28
    do
        for tc in 0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300
        do

            python new_test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
        done
    done
done
