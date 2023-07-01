DATA_DIR=$1

# #first train the model
for section in  objective_exam objective_results subjective assessment_and_plan #full 
do

                        python baselines/bart_summarization.py \
                            --tokenizer_name lidiya/bart-large-xsum-samsum \
                            --model_name_or_path lidiya/bart-large-xsum-samsum \
                            --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                            --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                            --test_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                            --text_column src \
                            --summary_column tgt \
                            --source_prefix " " \
                            --num_beams 5 \
                            --max_length 1024 \
                            --max_source_length 1024 \
                            --max_target_length 1024 \
                            --val_max_target_length 256 \
                            --val_min_target_length 128 \
                            --per_device_train_batch_size 1 \
                            --per_device_eval_batch_size 1 \
                            --num_train_epochs 15 \
                            --learning_rate 1e-5 \
                            --output_dir baselines/experiment/ablation_bart-large-xsum-samsum_valid_${section}/ \
                            --testing_dir baselines/experiment/ablation_bart-large-xsum-samsum_valid_${section}/ \
                            --seed 0890 
done


#evaluate the model with ablation
for trainset in  virtscribe_asr aci_asrcorr  # additional train set 
do
    for testset in test3 test1 test2  #valid 
    do
        for testset2 in aci_asr aci_asrcorr virtscribe_asr virtscribe_humantrans
        do
                for section in  objective_exam objective_results subjective assessment_and_plan #full 
                    do

                        #original
                        python baselines/bart_summarization.py \
                            --tokenizer_name lidiya/bart-large-xsum-samsum \
                            --model_name_or_path baselines/experiment/ablation_bart-large-xsum-samsum_valid_${section}/best_model \
                            --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                            --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                            --test_file $DATA_DIR/scr_experiment_data_json/${testset}_${testset2}_${section}.json \
                            --text_column src \
                            --summary_column tgt \
                            --source_prefix " " \
                            --num_beams 5 \
                            --max_length 1024 \
                            --max_source_length 1024 \
                            --max_target_length 1024 \
                            --val_max_target_length 256 \
                            --val_min_target_length 128 \
                            --per_device_train_batch_size 1 \
                            --per_device_eval_batch_size 1 \
                            --num_train_epochs 0 \
                            --learning_rate 1e-5 \
                            --output_dir baselines/experiment/ablation_bart-large-xsum-samsum_${testset}_${testset2}_${section}_train_${trainset}/ \
                            --testing_dir baselines/experiment/ablation_bart-large-xsum-samsum_${testset}_${testset2}_${section}_train_${trainset}/ \
                            --seed 0890 
                        
                        #bart-large-xsum-samsum - furthuer fine-tuning
                        python baselines/bart_summarization.py \
                            --tokenizer_name lidiya/bart-large-xsum-samsum \
                            --model_name_or_path baselines/experiment/ablation_bart-large-xsum-samsum_valid_${section}/best_model \
                            --train_file $DATA_DIR/scr_experiment_data_json/train_${trainset}_${section}.json \
                            --validation_file $DATA_DIR/scr_experiment_data_json/valid_${testset2}_${section}.json \
                            --test_file $DATA_DIR/scr_experiment_data_json/${testset}_${testset2}_${section}.json \
                            --text_column src \
                            --summary_column tgt \
                            --source_prefix " " \
                            --num_beams 5 \
                            --max_length 1024 \
                            --max_source_length 1024 \
                            --max_target_length 1024 \
                            --val_max_target_length 256 \
                            --val_min_target_length 128 \
                            --per_device_train_batch_size 1 \
                            --per_device_eval_batch_size 1 \
                            --num_train_epochs 3 \
                            --learning_rate 1e-5 \
                            --output_dir baselines/experiment/ablation_bart-large-xsum-samsum_${testset}_${testset2}_${section}_train_finetune3_${trainset}/ \
                            --testing_dir baselines/experiment/ablation_bart-large-xsum-samsum_${testset}_${testset2}_${section}_train_finetune3_${trainset}/ \
                            --seed 0890 
                    done
            done

    
        done 
done