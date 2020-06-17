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
    for depths in 10
    do
#for tc in 0 1 2 3 4 5 6 7 8 9 10 12 14 16
        for tc in 16
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
    for depths in 12
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10 12 14
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
    for depths in 14
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10 12
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
    for depths in 16
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10 12
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
    for depths in 20
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10
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
    for depths in 24
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10
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
    for depths in 28
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10
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
    for depths in 32
    do
        for tc in 0 1 2 3 4 5 6 7 8 9 10
        do
            for kkkkk in  0
            do
                python main_lin.py --depth=$depths --num_seg=$num_segs --width=8 --epochs=1000 --tc=$tc
            done
        done
    done
done
