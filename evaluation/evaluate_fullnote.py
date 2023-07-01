import sys
import json

import evaluate
import pandas as pd
import numpy as np
from UMLS_evaluation import *
from sectiontagger import SectionTagger
import torch

section_tagger = SectionTagger()

SECTION_DIVISIONS = [ 'subjective', 'objective_exam', 'objective_results', 'assessment_and_plan' ]

#set gpu core
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

def read_text( fn ) :
    texts = None
    if ".csv" in fn:
        temp_df=pd.read_csv(fn)
        texts=[temp_df["note"][ind] for ind in range(len(temp_df))]
    else:
        with open( fn ) as f :
            texts = f.readlines()
    return texts


def add_section_divisions( row ) :
    row[ 'src_len' ] = len( row[ 'dialogue' ].split() )
    for evaltype in [ 'reference', 'prediction' ] :
        text = row[ evaltype ]
        text_with_endlines = text.replace( '__lf1__', '\n' )
        detected_divisions = section_tagger.divide_note_by_metasections( text_with_endlines )
        for detected_division in detected_divisions :
            label, _, _, start, _, end = detected_division
            row[ '%s_%s' %( evaltype, label ) ] = text_with_endlines[ start:end ].replace( '\n', '__lf1__' )
    return row


if len( sys.argv ) < 3 :
    print( 'usage: python evaluate_fullnote.py <gold> <sys> <metadata-file>' )
    sys.exit(0)

#torch.cuda.empty_cache()
fn_gold = sys.argv[1]
fn_sys = sys.argv[2]
fn_metadata = [ sys.argv[3] if len( sys.argv )>3 else None ][0]

#read in reference/hyp files
references = read_text( fn_gold )
predictions = read_text( fn_sys  )
# handle edge cases
predictions = [str(s) for s in predictions]
print( 'gold path: %s [%s summuaries]' %(fn_gold, len(references) ) )
print( 'system path: %s [%s summuaries]' %(fn_sys, len(predictions) ) )

#read in metadata file - if none exists, just creates a dummy
if fn_metadata :
    df = pd.read_csv( fn_metadata )
    #print([len(df),len(references)])
    df[ 'reference' ] = references
    df[ 'prediction' ] = predictions
    df[ 'dialogue' ] = pd.read_csv(fn_gold)['dialogue']
else :
    #print(references)
    data = [ { 'id':ind, 'dataset':0, 'dialogue':'', 'reference':references[ind], 'prediction':predictions[ind]  } for ind in range( len( references ) ) ]
    df = pd.DataFrame( data )

#add section divisions
df = df.apply( lambda row: add_section_divisions( row ), axis=1 )
#fill in missing section divisions as empty string
df = df.fillna( '#####EMPTY#####' )
num_test = len( df )

######## CALCULATE PER INSTANCE SCORES ########
for division in SECTION_DIVISIONS :
    references.extend( df.get( 'reference_%s' %( division ), ['']*num_test ) )
    predictions.extend( df.get( 'prediction_%s' %( division ), ['']*num_test ) )

# sanity check, we should now have 5 x the original set (one for full note, 4 for the divisions)
assert len( references ) == len(df)*5, 'The number of expected references does not match expected'
assert len( predictions ) == len(df)*5, 'The number of expected predictions does not match expected'

results_umls_all=umls_score_group(references,predictions)
results_NER_all=umls_score_group(references,predictions,False)

results_rouge_all = evaluate.load('rouge').compute(references=references, predictions=predictions, use_aggregator=False)
#print( results_rouge_all )
#{'rouge1': [0.33333333333333326, 0.5217391304347826, 0.35294117647058826, 0.26666666666666666], 'rouge2': [0.125, 0.28571428571428575, 0.13333333333333333, 0.0], 'rougeL': [0.33333333333333326, 0.34782608695652173, 0.35294117647058826, 0.26666666666666666], 'rougeLsum': [0.33333333333333326, 0.34782608695652173, 0.35294117647058826, 0.26666666666666666]}

results_bertscore = evaluate.load('bertscore').compute(references=references, predictions=predictions, model_type='microsoft/deberta-xlarge-mnli')
# {'precision': [0.802226722240448, 0.8087180256843567, 0.7087132334709167, 0.7493441104888916], 'recall': [0.7557401657104492, 0.7800515294075012, 0.721121072769165, 0.7518222332000732], 'f1': [0.7782899141311646, 0.7941261529922485, 0.7148633003234863, 0.7505811452865601], 'hashcode': 'microsoft/deberta-xlarge-mnli_L40_no-idf_version=0.3.12(hug_trans=4.26.0)'}

results_bleurt = evaluate.load( 'bleurt', config_name='BLEURT-20').compute(references=references, predictions=predictions )
# # {'scores': [0.5392904281616211, 0.6561092734336853, 0.5172735452651978, 0.3984549045562744]}

results_all = { 
                "num_test":num_test,
                'ALL': { 'rouge1': np.mean( results_rouge_all['rouge1'][:num_test] ),
                          'rouge2': np.mean( results_rouge_all['rouge2'][:num_test] ),
                          'rougeL': np.mean( results_rouge_all['rougeL'][:num_test] ),
                          'rougeLsum': np.mean( results_rouge_all['rougeLsum'][:num_test] ),
                          'bertscore-precision': np.mean( results_bertscore['precision'][:num_test] ),
                          'bertscore-recall': np.mean( results_bertscore['recall'][:num_test] ),
                          'bertscore-f1': np.mean( results_bertscore['f1'][:num_test] ),
                          'bleurt': np.mean( results_bleurt['scores'][:num_test] ),
                        'umls': np.mean( results_umls_all[:num_test]),
                        'NER': np.mean( results_NER_all[:num_test]),
                        }
                }

# ######## CALCULATE PER-SUBSET SCORES ########
def select_values_by_indices( lst, indices ) :
    return [ lst[ind] for ind in indices ]

subsets = df[ 'dataset' ].unique().tolist()
for subset in subsets :
    indices = df[ df['dataset']==subset ].index.tolist()
    results_all[ 'dataset-%s' %subset ] = { 'rouge1': np.mean(  select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
                                            'rouge2': np.mean(  select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
                                            'rougeL': np.mean(  select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
                                            'rougeLsum': np.mean(  select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
                                            'bertscore-precision': np.mean(  select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
                                            'bertscore-recall': np.mean(  select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
                                            'bertscore-f1': np.mean(  select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
                                            'bleurt': np.mean(  select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
                                             'umls':   np.mean( select_values_by_indices( results_umls_all[:num_test], indices )),
                                            'NER':   np.mean( select_values_by_indices( results_NER_all[:num_test], indices )),
                                            }


######## CALCULATE PER-DIVISION SCORES ########
subsets = df[ 'dataset' ].unique().tolist()
for ind, division in enumerate( SECTION_DIVISIONS ) :
    start = (ind+1) * num_test
    end = (ind+2) * num_test
    results_all[ 'division-%s' %division ] = { 'rouge1': np.mean( results_rouge_all['rouge1'][start:end] ),
                                                'rouge2': np.mean( results_rouge_all['rouge2'][start:end] ),
                                                'rougeL': np.mean( results_rouge_all['rougeL'][start:end] ),
                                                'rougeLsum': np.mean( results_rouge_all['rougeLsum'][start:end] ),
                                                'bertscore-precision': np.mean( results_bertscore['precision'][start:end] ),
                                                'bertscore-recall': np.mean( results_bertscore['recall'][start:end] ),
                                                'bertscore-f1': np.mean( results_bertscore['f1'][start:end] ),
                                                'bleurt': np.mean( results_bleurt['scores'][start:end] ),
                                                'umls': np.mean( results_umls_all[start:end]),
                                                'NER': np.mean( results_NER_all[start:end]),
                                                }


######## CALCULATE PER-LENGTH SCORES (bigger than 512 vs not) ########
df_shortsrc = df[ df['src_len']<=512 ]
if len( df_shortsrc ) > 0 :
    indices = df_shortsrc[:num_test].index.tolist()
    results_all[ 'shorter-src' ] = { 'rouge1': np.mean(  select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
                                                'rouge2': np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
                                                'rougeL': np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
                                                'rougeLsum': np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
                                                'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
                                                'bertscore-recall': np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
                                                'bertscore-f1': np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
                                                'bleurt': np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
                                                'umls':   np.mean( select_values_by_indices( results_umls_all[:num_test], indices )),
                                                'NER':   np.mean( select_values_by_indices( results_NER_all[:num_test], indices )),
                                                }
df_longsrc = df[ df['src_len']>512 ]
if len( df_longsrc ) > 0 :
    indices = df_longsrc[:num_test].index.tolist()
    results_all[ "longer-src (support:{})".format(len(df_longsrc)) ] = { 'rouge1': np.mean(  select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
                                                'rouge2': np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
                                                'rougeL': np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
                                                'rougeLsum': np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
                                                'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
                                                'bertscore-recall': np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
                                                'bertscore-f1': np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
                                                'bleurt': np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
                                                'umls':   np.mean( select_values_by_indices( results_umls_all[:num_test], indices )),
                                                'NER':   np.mean( select_values_by_indices( results_NER_all[:num_test], indices )),
                                            
                                                }

## adding for each meta information
##patient_gender,patient_age,patient_firstname,patient_familyname,cc,2nd_complaints
for meta_type in ["patient_gender","cc","2nd_complaints"]:
    if meta_type in df:
        values=set(list(df[meta_type]))
        for value in values:
            df_meta = df[ df[meta_type] == value ]
            if len( df_meta  ) > 0 :
                indices = df_meta [:num_test].index.tolist()
                results_all[ meta_type+"-{}(support:{})".format(value,len(df_meta)) ] = { 'rouge1': np.mean(  select_values_by_indices( results_rouge_all['rouge1'][:num_test], indices ) ),
                                                            'rouge2': np.mean( select_values_by_indices( results_rouge_all['rouge2'][:num_test], indices ) ),
                                                            'rougeL': np.mean( select_values_by_indices( results_rouge_all['rougeL'][:num_test], indices ) ),
                                                            'rougeLsum': np.mean( select_values_by_indices( results_rouge_all['rougeLsum'][:num_test], indices ) ),
                                                            'bertscore-precision': np.mean( select_values_by_indices( results_bertscore['precision'][:num_test], indices ) ),
                                                            'bertscore-recall': np.mean( select_values_by_indices( results_bertscore['recall'][:num_test], indices ) ),
                                                            'bertscore-f1': np.mean( select_values_by_indices( results_bertscore['f1'][:num_test], indices ) ),
                                                            'bleurt': np.mean( select_values_by_indices( results_bleurt['scores'][:num_test], indices ) ),
                                                            'umls':   np.mean( select_values_by_indices( results_umls_all[:num_test], indices )),
                                                            'NER':   np.mean( select_values_by_indices( results_NER_all[:num_test], indices )),
                                                            }

###### OUTPUT TO JSON FILE ########
json_object = json.dumps( results_all, indent=4 )
fn_out = 'results/{}.json'.format(fn_sys.split("/")[-1].split(".")[0])
with open( fn_out, 'w' ) as f :
    f.write( json_object )