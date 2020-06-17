


for num_segs in 20
do
    for depths in 16
    do
        for tc in 0 20 40 60 80 100 120 140
        do

            python test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 20
    do
        for tc in 0 20 40 60 80 100 120 140 160
        do

            python test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 24
    do
        for tc in 0 20 40 60 80 100 120 140 160 180 200
        do

            python test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
        done
    done
done


for num_segs in 20
do
    for depths in 32
    do
        for tc in 0 20 40 60 80 100 120 140 160 180 200 220 240 260
        do

            python test.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
        done
    done
done