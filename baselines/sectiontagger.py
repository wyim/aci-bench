import re
import sys
from glob import glob
import json
from collections import OrderedDict
import os

#a colon version asks for :, but also will give a variation of no colon on it's own seperate line
#to also inlude a no colon version (but allow : version to match greedily first) supply the colon version prior to a no colon version.

subjective_subsections = OrderedDict()
subjective_subsections[ 'cc' ] = [ 'cc :', 'chief complaint :', 'reason for visit :',"CHIEF COMPLAINT" ]
subjective_subsections[ 'hpi' ] = [ 'history :', 'history of present illness :', 
                                   'history of present illness', 'hpi :', 'hpi', 'hpi notes :', 
                                   'interval history :', 'interval hx :', 'subjective :',
                                   "HISTORY OF PRESENT ILLNESS"]
subjective_subsections[ 'ros' ] = [ 'ros :', 'review of system :', 'review of systems :' ]
subjective_subsections[ 'other_histories' ] = [ 'SOCIAL HISTORY',"PAST HISTORY"]

objectiveexam_subsections = OrderedDict()
objectiveexam_subsections[ 'pe' ] = [ 'physical exam :', 'physical examination :', 'pe :', 
                                     'physical findings :', 'examination :', 'exam :',
                                     "PHYSICAL EXAMINATION","PHYSICAL EXAM",]
objectiveexam_subsections[ 'vitals' ] = ['VITALS REVIEWED',]

objectiveresults_subsections = OrderedDict()
objectiveresults_subsections[ 'findings' ] = [ 'results :', 'findings :',"RESULTS",
                                              ]

ap_subsections = OrderedDict()
ap_subsections[ 'assessment' ] = ['assessment :', 'a:' ]
ap_subsections[ 'plan' ] = [ 'plan :', 'plan of care :', 'p:', 'medical decision-making plan :', 'summary plan' ]
ap_subsections[ 'ap' ] = [ 'ap :', 'a / p :', 'assessment and plan :', 'assessment & plan :',
                          'disposition / plan :',
                            "ASSESSMENT AND PLAN", ]

sectcat2subsections = OrderedDict()
sectcat2subsections[ 'subjective' ] = subjective_subsections
sectcat2subsections[ 'objective_exam' ] = objectiveexam_subsections
sectcat2subsections[ 'objective_results' ] = objectiveresults_subsections
sectcat2subsections[ 'assessment_and_plan' ] = ap_subsections


#default name when there is no sectionheader
NOSECTIONHEADER = 'default'
#map between subsectionheader to it's parent section
subsectionheader2section = {}
for sh, sshdicts in sectcat2subsections.items() :
    for ssh, lst in sshdicts.items() :
        subsectionheader2section[ ssh ] = sh
subsectionheader2section[ NOSECTIONHEADER ] = NOSECTIONHEADER


class SectionTagger() :
    
    def __init__( self, sectcat2subsections=sectcat2subsections ) :
        #compile regex for each section
        self.sectcat2subsections = sectcat2subsections
        self.compileregexes()
    
    def compileregexes( self ) :
        self.subsect2regex = {}
        for _, sectcat2subsections in self.sectcat2subsections.items() :
            for subsect, vlst in sectcat2subsections.items() :
                self.subsect2regex[ subsect ] = self._compile_regexexpression( vlst )
    
    def _compile_regexexpression( self, vlst ) :
        expressions = []
        otherexps = []
        for exp in vlst :
            exp2 = '(' + re.escape( exp ).replace( '\ ', '\s*' ) +')'
            expressions.append( exp2 )
            if exp[-1] == ':' :
                #allow without : if the line is empty
                exp2 = '(' + re.escape( exp[:-1] ).replace( '\ ', '\s*' ) +')'
                otherexps.append( exp2 )
        
        patt = '\s*(?P<sectionheader1>' + '|'.join( expressions ) + ').*'
        if len( otherexps ) > 0 :
            pattott = '\s*(?P<sectionheader2>' + '|'.join( otherexps ) + ')\s*$'
            return '(' + patt + '|' + pattott + ')'
        
        return patt

    def tag_sectionheaders( self, text ) :
        """
        Input : text
        Return: sections : list of tupples ( subsectionheader, linenum, char_start, char_end  )
        """
        #assumes text is sentence tokenize according to lines
        offset = 0
        subsects = []
        for linenum, line in enumerate( text.split( '\n' ) ) :
            for subsect, rgx in self.subsect2regex.items() :
                m = re.match( rgx, line, re.IGNORECASE )
                if m :
                    secthlib = m.groupdict()
                    secthpattname = [ 'sectionheader1' if ( secthlib['sectionheader1'] is not None ) else 'sectionheader2' ][0]
                    subsects.append( ( subsect, linenum, offset, offset+m.end( secthpattname ) ) )

            offset += len( line ) + 1
        return subsects
    
    def tag_sections( self, text ) :
        """
        Given text, return list of tuples:
        ( sectionheader, subsectionheader, subsectionheader_line_start, subsectheader_start, subsectheader_end, subsectionend )

        This function will assign the overall section, and also mark the end of a subsection.
        If multiple subsections get marked per line, we will take the first occurence according to the usual order.
        End of subsection is demarked by the beginning of a new subsectionheader.
        """
        #go through regex to get the sectionheaders
        subsects = self.tag_sectionheaders( text )

        #save lineno2tuple, if there is a lineno conflict then take the first occurence according to our ordereddicts
        linenum2tuple = {}
        for subsect in subsects :
            linenum= subsect[ 1 ]
            if linenum not in linenum2tuple :
                linenum2tuple[ linenum ] = subsect
        
        #enumerate through lines, track current section
        previoussection = NOSECTIONHEADER
        currentsection = NOSECTIONHEADER
        
        sectionlist = []
        offset = 0
        
        #check if first line 
        if 0 in linenum2tuple :
            prevsectionheadertuple = linenum2tuple[ 0 ]
        else :
            prevsectionheadertuple = ( NOSECTIONHEADER, 0, 0, 0 )

        lines = text.split( '\n' )
        offset = len( lines[0] ) + 1
        for linenum, line in enumerate( lines[1:], 1 ) :

            if linenum in linenum2tuple :
                shtuple = linenum2tuple[ linenum ]

                prevsection = subsectionheader2section[ prevsectionheadertuple[0] ]
                sectionlist.append( [prevsection] + list( prevsectionheadertuple ) + [ offset ] )
                prevsectionheadertuple = shtuple
            offset += len( line ) + 1
    
        prevsection = subsectionheader2section[ prevsectionheadertuple[0] ]
        sectionlist.append( [prevsection] + list( prevsectionheadertuple ) + [ len( text ) ] )

        return sectionlist
    

    def divide_note_by_metasections( self, text ) :

        detected_sections = self.tag_sections( text )

        #if starts with no sectionheader, we shall just assign as subjective
        if ( len( detected_sections ) > 0 ) and detected_sections[0][0] == NOSECTIONHEADER :
            detected_sections[0][0] = 'subjective'

        #only keep one of each divison type
        meta_sections = [ None, None, None, None ]
        for section in detected_sections :
            if ( section[0] == 'subjective' ) and meta_sections[0] is None :
                meta_sections[0] = section
            if ( section[0] == 'objective_exam' ) and meta_sections[1] is None :
                meta_sections[1] = section
            if ( section[0] == 'objective_results' ) and meta_sections[2] is None :
                meta_sections[2] = section
            if ( section[0] == 'assessment_and_plan' ) and meta_sections[3] is None :
                meta_sections[3] = section
        
        #filter out none, and order by appearance
        meta_sections = [ x for x in meta_sections if x is not None ]
        meta_sections = sorted( meta_sections, key= lambda x: x[3])
        
        #adjust the end of section text offset
        # e.g. ['subjective', 'cc', 0, 0, 15, 30]
        for ind, section in enumerate( meta_sections[1:], start=1 ) :
            meta_sections[ ind-1 ][-1] = section[ -3 ]

        if len( meta_sections ) > 0 :
            meta_sections[ -1 ][-1] = len( text )

        return meta_sections

import re
import sys
from glob import glob
import json
from collections import OrderedDict

#a colon version asks for :, but also will give a variation of no colon on it's own seperate line
#to also inlude a no colon version (but allow : version to match greedily first) supply the colon version prior to a no colon version.

subjective_subsections = OrderedDict()
subjective_subsections[ 'cc' ] = [ 'cc :', 'chief complaint :', 'reason for visit :',"CHIEF COMPLAINT" ]
subjective_subsections[ 'hpi' ] = [ 'history :', 'history of present illness :', 
                                   'history of present illness', 'hpi :', 'hpi', 'hpi notes :', 
                                   'interval history :', 'interval hx :', 'subjective :',
                                   "HISTORY OF PRESENT ILLNESS"]
subjective_subsections[ 'ros' ] = [ 'ros :', 'review of system :', 'review of systems :' ]
subjective_subsections[ 'other_histories' ] = [ 'SOCIAL HISTORY',"PAST HISTORY"]

objectiveexam_subsections = OrderedDict()
objectiveexam_subsections[ 'pe' ] = [ 'physical exam :', 'physical examination :', 'pe :', 
                                     'physical findings :', 'examination :', 'exam :',
                                     "PHYSICAL EXAMINATION","PHYSICAL EXAM",]
objectiveexam_subsections[ 'vitals' ] = ['VITALS REVIEWED',]

objectiveresults_subsections = OrderedDict()
objectiveresults_subsections[ 'findings' ] = [ 'results :', 'findings :',"RESULTS",
                                              ]

ap_subsections = OrderedDict()
ap_subsections[ 'assessment' ] = ['assessment :', 'a:' ]
ap_subsections[ 'plan' ] = [ 'plan :', 'plan of care :', 'p:', 'medical decision-making plan :', 'summary plan' ]
ap_subsections[ 'ap' ] = [ 'ap :', 'a / p :', 'assessment and plan :', 'assessment & plan :',
                          'disposition / plan :',
                            "ASSESSMENT AND PLAN", ]

sectcat2subsections = OrderedDict()
sectcat2subsections[ 'subjective' ] = subjective_subsections
sectcat2subsections[ 'objective_exam' ] = objectiveexam_subsections
sectcat2subsections[ 'objective_results' ] = objectiveresults_subsections
sectcat2subsections[ 'assessment_and_plan' ] = ap_subsections


#default name when there is no sectionheader
NOSECTIONHEADER = 'default'
#map between subsectionheader to it's parent section
subsectionheader2section = {}
for sh, sshdicts in sectcat2subsections.items() :
    for ssh, lst in sshdicts.items() :
        subsectionheader2section[ ssh ] = sh
subsectionheader2section[ NOSECTIONHEADER ] = NOSECTIONHEADER


class SectionTagger() :
    
    def __init__( self, sectcat2subsections=sectcat2subsections ) :
        #compile regex for each section
        self.sectcat2subsections = sectcat2subsections
        self.compileregexes()
    
    def compileregexes( self ) :
        self.subsect2regex = {}
        for _, sectcat2subsections in self.sectcat2subsections.items() :
            for subsect, vlst in sectcat2subsections.items() :
                self.subsect2regex[ subsect ] = self._compile_regexexpression( vlst )
    
    def _compile_regexexpression( self, vlst ) :
        expressions = []
        otherexps = []
        for exp in vlst :
            exp2 = '(' + re.escape( exp ).replace( '\ ', '\s*' ) +')'
            expressions.append( exp2 )
            if exp[-1] == ':' :
                #allow without : if the line is empty
                exp2 = '(' + re.escape( exp[:-1] ).replace( '\ ', '\s*' ) +')'
                otherexps.append( exp2 )
        
        patt = '\s*(?P<sectionheader1>' + '|'.join( expressions ) + ').*'
        if len( otherexps ) > 0 :
            pattott = '\s*(?P<sectionheader2>' + '|'.join( otherexps ) + ')\s*$'
            return '(' + patt + '|' + pattott + ')'
        
        return patt

    def tag_sectionheaders( self, text ) :
        """
        Input : text
        Return: sections : list of tupples ( subsectionheader, linenum, char_start, char_end  )
        """
        #assumes text is sentence tokenize according to lines
        offset = 0
        subsects = []
        for linenum, line in enumerate( text.split( '\n' ) ) :
            for subsect, rgx in self.subsect2regex.items() :
                m = re.match( rgx, line, re.IGNORECASE )
                if m :
                    secthlib = m.groupdict()
                    secthpattname = [ 'sectionheader1' if ( secthlib['sectionheader1'] is not None ) else 'sectionheader2' ][0]
                    subsects.append( ( subsect, linenum, offset, offset+m.end( secthpattname ) ) )

            #adding special case of impression as assessment (but don't want lower case versions to show up,
            # as our dataset has impression as part of results )
            m = re.match( '\s*IMPRESSION', line )
            if m :
                subsects.append( ( 'impression', linenum, offset, offset+m.end() ) )

            offset += len( line ) + 1
        return subsects
    
    def tag_sections( self, text ) :
        """
        Given text, return list of tuples:
        ( sectionheader, subsectionheader, subsectionheader_line_start, subsectheader_start, subsectheader_end, subsectionend )

        This function will assign the overall section, and also mark the end of a subsection.
        If multiple subsections get marked per line, we will take the first occurence according to the usual order.
        End of subsection is demarked by the beginning of a new subsectionheader.
        """
        #go through regex to get the sectionheaders
        subsects = self.tag_sectionheaders( text )

        #save lineno2tuple, if there is a lineno conflict then take the first occurence according to our ordereddicts
        linenum2tuple = {}
        for subsect in subsects :
            linenum= subsect[ 1 ]
            if linenum not in linenum2tuple :
                linenum2tuple[ linenum ] = subsect
        
        #enumerate through lines, track current section
        previoussection = NOSECTIONHEADER
        currentsection = NOSECTIONHEADER
        
        sectionlist = []
        offset = 0
        
        #check if first line 
        if 0 in linenum2tuple :
            prevsectionheadertuple = linenum2tuple[ 0 ]
        else :
            prevsectionheadertuple = ( NOSECTIONHEADER, 0, 0, 0 )

        lines = text.split( '\n' )
        offset = len( lines[0] ) + 1
        for linenum, line in enumerate( lines[1:], 1 ) :

            if linenum in linenum2tuple :
                shtuple = linenum2tuple[ linenum ]

                if prevsectionheadertuple[0] == 'impression' :
                    prevsection = 'assessment_and_plan'
                else :
                    prevsection = subsectionheader2section[ prevsectionheadertuple[0] ]

                sectionlist.append( [prevsection] + list( prevsectionheadertuple ) + [ offset ] )
                prevsectionheadertuple = shtuple
            offset += len( line ) + 1
    
        #adding special case of impression as assessment
        if prevsectionheadertuple[0] == 'impression' :
            prevsection = 'assessment_and_plan'
        else :
            prevsection = subsectionheader2section[ prevsectionheadertuple[0] ]

        sectionlist.append( [prevsection] + list( prevsectionheadertuple ) + [ len( text ) ] )

        return sectionlist
    

    def divide_note_by_metasections( self, text ) :

        detected_sections = self.tag_sections( text )

        #if starts with no sectionheader, we shall just assign as subjective
        if ( len( detected_sections ) > 0 ) and detected_sections[0][0] == NOSECTIONHEADER :
            detected_sections[0][0] = 'subjective'

        #only keep one of each divison type
        meta_sections = [ None, None, None, None ]
        for section in detected_sections :
            if ( section[0] == 'subjective' ) and meta_sections[0] is None :
                meta_sections[0] = section
            if ( section[0] == 'objective_exam' ) and meta_sections[1] is None :
                meta_sections[1] = section
            if ( section[0] == 'objective_results' ) and meta_sections[2] is None :
                meta_sections[2] = section
            if ( section[0] == 'assessment_and_plan' ) and meta_sections[3] is None :
                meta_sections[3] = section
        
        #filter out none, and order by appearance
        meta_sections = [ x for x in meta_sections if x is not None ]
        meta_sections = sorted( meta_sections, key= lambda x: x[3])
        
        #adjust the end of section text offset
        # e.g. ['subjective', 'cc', 0, 0, 15, 30]
        for ind, section in enumerate( meta_sections[1:], start=1 ) :
            meta_sections[ ind-1 ][-1] = section[ -3 ]

        if len( meta_sections ) > 0 :
            meta_sections[ -1 ][-1] = len( text )

        return meta_sections


if __name__ == "__main__" :
    #
    datasets=["train","valid","clinicalnlp_taskB_test1","clinicalnlp_taskC_test2","clef_taskC_test3"]
    data_dirs=["./data/challenge_data_json/"] #,"./data/src_experiment_data_json/"

    # datasets=[file.split("/")[-1].split(".")[0] for file in glob("./data/src_experiment_data_json/**")]
    # data_dirs=["./data/src_experiment_data_json/"] #,"./data/src_experiment_data_json/"
    for dataset in datasets:
        for data_dir in data_dirs:
        
            fn=data_dir+dataset+".json"
            assert os.path.isfile(fn),fn
            dic=json.loads(open(fn,encoding="utf-8").read())['data']
            
            with open(fn.replace(".json","_full.json"),"w",encoding="utf-8") as f:
                    json.dump({"data":dic},f,indent=4)

            output={
                    "subjective":[],
                    "objective_exam":[],
                    "objective_results":[],
                    "assessment_and_plan":[]
                    }
            
            count=[0]*4
            for instance in dic:
                #output the default
                for key in output:
                    output[key].append({"src":instance["src"],
                                        "tgt":"",
                                        "file":instance["file"]+"-"+key})
                    
                section_tagger = SectionTagger()
                secttups = section_tagger.tag_sections(instance["tgt"] )

                ent_id = 0
                
                texts = []

                #filling all sections
                for i,secttup in enumerate(secttups) :
                    
                    sectionheader, subsectionheader, _, start, _, end = secttup

                    if sectionheader!='default':
                        output[sectionheader][-1]["tgt"]+=instance["tgt"][int(start):int(end)]
                for key in output:
                    if not output[key][-1]["tgt"]:
                        output[key][-1]["tgt"]=key+": #####EMPTY#####.\n"
                        count[list(output.keys()).index(key)]+=1
            
            #write_output
            for key in output:
                with open(fn.replace(".json","_"+key+".json"),"w",encoding="utf-8") as f:
                    json.dump({"data":output[key]},f,indent=4)