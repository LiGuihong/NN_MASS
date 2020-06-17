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

for num_segs in 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20
do
    python random_main.py  --num_seg=$num_segs --width=8 --epochs=600 
done
