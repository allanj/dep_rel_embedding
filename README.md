## Pretraining GNN for better dependency relation representation


### Requirements
* PyTorch 1.4 (Also tested on PyTorch 1.3)
* Python 3.6

### Dataset Format

I have uploaded the preprocessed `Catalan` and `Spanish` datasets. (Please contact me with your license if you need the preprocessed OntoNotes dataset.)
If you have a new dataset, please make sure we follow the CoNLL-X format and we put the entity label at the end.
The sentence below is an example.
Note that we only use the columns for *word*, *dependency head index*, *dependency relation label* and the last *entity label*.
```
1	Brasil	_	n	n	_	2	suj	_	_	B-org
2	buscará	_	v	v	_	0	root	_	_	O
3	a_partir_de	_	s	s	_	2	cc	_	_	O
4	mañana	_	n	n	_	3	sn	_	_	O
5	,	_	f	f	_	6	f	_	_	B-misc
6	viernes	_	w	w	_	4	sn	_	_	I-misc
7	,	_	f	f	_	6	f	_	_	I-misc
8	el	_	d	d	_	9	spec	_	_	O
9	pase	_	n	n	_	2	cd	_	_	O
```
Entity labels follow the `IOB` tagging scheme and will be converted to `IOBES` in this codebase.




