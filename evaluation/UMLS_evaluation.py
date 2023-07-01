import os

PROJECT_DIR = os.path.dirname( os.path.dirname( __file__ ) )

#define the fact-based extractor
quickumls_fp = PROJECT_DIR+ "/resources/des"
#the window size for transitioning
WINDOW_SIZE=5
COUNT_THRESHOLD=50
ENCODING="utf-8"

import spacy
nlp = spacy.load("en_ner_bc5cdr_md")


from semantics import SEMANTICS

from quickumls import QuickUMLS
matcher = QuickUMLS(quickumls_fp,window=WINDOW_SIZE,threshold=1,accepted_semtypes=SEMANTICS)

def get_matches(text,use_umls=True):
    concepts={}
    cui_list=[]
    if use_umls:
        matches=matcher.match(text, ignore_syntax=True)
        for match in matches:
            for m in match:
                if m['cui'] not in concepts.get(m['term'],[]):
                    concepts[m['term']]=concepts.get(m['term'],[])+[m['cui']]
                    cui_list.append(m['cui'])
    else:
        doc = nlp(text)
        #linker = nlp.get_pipe("scispacy_linker")
        for ent in doc.ents:
            key=(ent.text.lower(),ent.label_)
            if ent.text not in concepts.get(key,[]):
                    concepts[key]=concepts.get(key,[])+[ent.text]
                    cui_list.append(ent.text)
    return concepts,cui_list

def umls_score_individual(reference,prediction,use_umls=True):
    true_concept,true_cuis=get_matches(reference,use_umls)
    pred_concept,pred_cuis=get_matches(prediction,use_umls)
    try:
        num_t=0
        for key in true_concept:
            for cui in true_concept[key]:
                if cui in pred_cuis:
                    num_t+=1
                    break
        
        precision=num_t*1.0/len(pred_concept.keys())
        recall=num_t*1.0/len(true_concept.keys())
        F1=2*(precision*recall)/(precision+recall)
        return F1
    except:
        return 0

def umls_score_group(references,predictions,use_umls=True):   
    return [umls_score_individual(reference,prediction,use_umls) for reference,prediction in zip(references,predictions)]
