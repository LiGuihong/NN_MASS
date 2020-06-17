#for num_segs in 10
#do
#    for depths in 5
#    do
#        for tc in 0 2 4 6 8 10 12 14 16 18 20 22 24
#        do 
#            for kkkkk in  0 0 0 0 0
#            do
#                python main_v4.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=600 --tc=$tc
#            done
#        done
#    done
#done

for num_segs in 20
do
    for depths in 6
    do
        for tc in 0 4 8 12 16 18 20 22 24 26 28 30 32 
        do
            for kkkkk in  0
            do
                python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
            done
        done
    done
done

for num_segs in 20
do
    for depths in 8
    do
        for tc in 0 4 6 8 10 12 14 16 18 20 24 28 32 36 40 44 48
        do
            for kkkkk in  0
            do
                python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1500 --tc=$tc
            done
        done
    done
done

for num_segs in 20
do
    for depths in 10
    do
        for tc in 0 2 4 6 8 10 12 14 16 20 24 28 32 36 40 44 48
        do
            for kkkkk in  0
            do
                python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1500 --tc=$tc
            done
        done
    done
done

for num_segs in 20
do
    for depths in 12
    do
        for tc in 0 2 4 6 8 10 12 14 16 20 24 28 32 36 40 44 48
        do
            for kkkkk in  0 0 0 0 0
            do
                python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1500 --tc=$tc
            done
        done
    done
done