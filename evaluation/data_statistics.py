#%%
import re
import pandas as pd
import numpy as np

#%%
import sys

sys.path.append( '../baselines/' )
#%%
from sectiontagger import SectionTagger
section_tagger = SectionTagger()
SECTION_DIVISIONS = [ 'subjective', 'objective_exam', 'objective_results', 'assessment_and_plan' ]
#%%

import spacy
nlp = spacy.load("en_core_sci_lg")

#%%
indir = '../data/challenge_data'

splits = [ 'train', 'valid', 'clinicalnlp_taskB_test1', 'clinicalnlp_taskC_test2', 'clef_taskC_test3' ]
fields = [ 'src_len', 'src_turnlen', 'note_len', 'note_sentlen' ]

def add_lengths( row ) :
    row[ 'src_len' ] = len( row['dialogue'].split() )

    row[ 'src_turnlen' ] = len( re.findall( r"\[\S+\]", row['dialogue'] ) )
    
    doc = nlp( row['note'] )
    row[ 'note_len' ] = len( doc )
    row[ 'note_sentlen' ] = len( list( doc.sents ) )

    detected_divisions = section_tagger.divide_note_by_metasections( row['note'] )
    detected_divisions = set( [ x[0] for x in detected_divisions ])
    for div in SECTION_DIVISIONS :
        row[ div ] = [ 1 if div in detected_divisions else 0 ][0]
    
    return row

#%%
for split in splits :

    print( '===================================' )
    print( '%s' %split )
    print( '===================================' )

    df = pd.read_csv( '%s/%s.csv' %( indir, split ) )
    df = df.apply( lambda row: add_lengths( row ), axis=1 )

    for field in fields :
        print( '--------------' )
        print( '%s' %field )
        print( '--------------' )
        print( df[ [ 'dataset', field ] ].groupby( by='dataset' ).agg([ "count", "mean", "std", "min", "max"]) )
        print( '--------------' )
        print( df[ field ].agg([ "count", "mean", "std", "min", "max"]) )
        print( '--------------' )
        print()

    for field in SECTION_DIVISIONS :
        print( '--------------' )
        print( '%s' %field )
        print( '--------------' )
        print( df[ [ 'dataset', field ] ].groupby( by='dataset' ).sum() )
        print( df[ field ].sum() )


