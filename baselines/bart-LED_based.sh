DATA_DIR=$1

for testset in  clinicalnlp_taskB_test1 clinicalnlp_taskC_test2 clef_taskC_test3 # 
do
        for section in full objective_exam objective_results subjective  assessment_and_plan #
            do
            
                #BioBART
                python baselines/bart_summarization.py \
                    --model_name_or_path GanjinZero/biobart-v2-large \
                    --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                    --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                    --test_file $DATA_DIR/challenge_data_json/${testset}_${section}.json \
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
                    --output_dir baselines/experiments/BioBART_${testset}_${section}/ \
                    --testing_dir baselines/experiments/BioBART_${testset}_${section}/ \
                    --seed 0890 

                # BART-Large
                python baselines/bart_summarization.py \
                    --model_name_or_path facebook/bart-large \
                    --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                    --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                    --test_file $DATA_DIR/challenge_data_json/${testset}_${section}.json \
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
                    --output_dir baselines/experiments/BART_large_${testset}_${section}/ \
                    --testing_dir baselines/experiments/BART_large_${testset}_${section}/ \
                    --seed 0890 

                #bart-large-xsum-samsum
                python baselines/bart_summarization.py \
                    --model_name_or_path lidiya/bart-large-xsum-samsum \
                    --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                    --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                    --test_file $DATA_DIR/challenge_data_json/${testset}_${section}.json \
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
                    --num_train_epochs 1 \
                    --learning_rate 1e-5 \
                    --output_dir baselines/experiments/bart-large-xsum-samsum_${testset}_${section}/ \
                    --testing_dir baselines/experiments/bart-large-xsum-samsum_${testset}_${section}/ \
                    --seed 0890 

                # LED
                python baselines/longformer_summarization.py \
                    --model_name_or_path allenai/led-large-16384 \
                    --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                    --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                    --test_file $DATA_DIR/challenge_data_json/${testset}_${section}.json \
                    --text_column src \
                    --summary_column tgt \
                    --source_prefix " " \
                    --num_beams 5 \
                    --max_length 2048 \
                    --max_source_length 2048 \
                    --max_target_length 1024 \
                    --val_max_target_length 1024  \
                    --val_min_target_length 384 \
                    --global_attention 1024\
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 1 \
                    --num_train_epochs 15 \
                    --learning_rate 1e-5 \
                    --output_dir baselines/experiments/LED_${testset}_${section}/ \
                    --seed 0890
                
                # LED-pubmed
                python baselines/longformer_summarization.py \
                    --model_name_or_path patrickvonplaten/led-large-16384-pubmed \
                    --train_file $DATA_DIR/challenge_data_json/train_${section}.json \
                    --validation_file $DATA_DIR/challenge_data_json/valid_${section}.json \
                    --test_file $DATA_DIR/challenge_data_json/${testset}_${section}.json \
                    --text_column src \
                    --summary_column tgt \
                    --source_prefix " " \
                    --num_beams 5 \
                    --max_length 2048 \
                    --max_source_length 2048 \
                    --max_target_length 1024 \
                    --val_max_target_length 1024  \
                    --val_min_target_length 384 \
                    --global_attention 1024\
                    --per_device_train_batch_size 1 \
                    --per_device_eval_batch_size 1 \
                    --num_train_epochs 15 \
                    --learning_rate 1e-5 \
                    --output_dir baselines/experiments/LED_pubmed_${testset}_${section}/ \
                    --seed 0890 
            done
done

