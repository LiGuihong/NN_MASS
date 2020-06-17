for num_segs in 30 40
do
    for depths in 10
    do
        for shortcut_nums in 0 5 10 15 20 25 30 35 36
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs --width=30 --epochs=2000 --shortcut_num=$shortcut_nums
        done
    done
done



for num_segs in 10 20 30 40
do
    for depths in 20
    do
        for shortcut_nums in 15 20 25 30 35
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs --width=15 --shortcut_num=$shortcut_nums
        done
    done
done



for num_segs in 10 20 30 40
do
    for depths in 30
    do
        for shortcut_nums in 15 20 25 30 35
        do

            python main_v3.py --depth=$depths --num_seg=$num_segs --width=10 --shortcut_num=$shortcut_nums
        done
    done
done

