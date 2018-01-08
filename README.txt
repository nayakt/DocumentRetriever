Dependencies:
----------------
1) Python3.5
2) Numpy
3) NLTK

Run Indexer
--------------

python3.5 indexer.py 
		-docs "directory of corpus which to be indexed"			// mandatory argument
		-index "directory where index will be stored" 			// mandatory argument
		-mode "0 for new index, 1 for updating old index" 		// optional, default 0
		-max_size "max tokens in file" 					// optional, default 200 
		-min_size "minimum tokens in file"				// optional, default 40

Example: python3.5 indexer.py -docs /home/nayak/Work/TREC/trec-data/trec-segmented-130/trec2001 -index /home/nayak/Work/QAFocusedIR/Experiment/trec-2001/index/  -mode 0 -max_size 200 -min_size 40


Run Searcher
---------------

python3.5 searcher.py 
		-index "Index directory" 				// mandatory argument
		-qs "question file" 					// mandatory argument
		-output "rank list will be written in this file" 	// mandatory argument
		-top 'number of top docs to be retrieved' 		// optional, default 10
		-error "any error will be written to this file"		// mandatory argument

Example: python3.5 searcher.py -index /home/nayak/Work/QAFocusedIR/Experiment/trec-2001/index/ -qs /home/nayak/Work/QAFocusedIR/Experiment/trec-2001/mod_questions.txt -output /home/nayak/Work/QAFocusedIR/Experiment/trec-2001/rank-list.txt -top 500 -error /home/nayak/Work/QAFocusedIR/Experiment/trec-2001/error.txt


