for num_segs in 20
do
    for depths in 10
    do
        for shortcut_nums in 0 5 10 15 20 25 30 35 36 36 0 5 10 15 20 25 30 35 36 36 0 5 10 15 20 25 30 35 36 36
        do

            python main_v4.py --depth=$depths --num_seg=$num_segs --width=20 --epochs=1000 --shortcut_num=$shortcut_nums
        done
    done
done

