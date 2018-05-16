# Data

## Annotation

For details on annotation, see Han-Chin Shing, Suraj Nair, Ayah Zirikly, Meir Friedenberg, Hal Daumé III and Philip Resnik (2018), "Expert, Crowdsourced, and Machine Assessment of Suicide Risk via Online Postings", submitted for publication.

Two sets of annotations are included here in the reddit_annotations directory, `crowd.csv` and `expert.csv`. Note that these files contain data pertaining to two disjoint sets of users (meaning that there is no user who appears in both).

`crowd.csv` is intended to serve as training/development data.

`expert.csv` is intended to serve as held-out test data.

Users are identified by numeric user_ids. By convention, the user_id is a negative number for users who did *not* post on SuicideWatch, i.e. control users.  The user_id is what links annotated data in this directory to the actual postings in ../reddit_posts.


### Crowd

`crowd.csv`: Crowdsourced annotation conducted using CrowdFlower. This file contains annotations for 621 users who posted on SuicideWatch and 621 control users.

Column B (raw_label) is the "raw" label, meaning the consensus label assigned by CrowdFlower based on the crowdsourcers' labeling. Possible values include *a*, *b*, *c*, *d* or *None*. *a* means *No Risk*, *b* means *Low Risk*, *c* means *Moderate Risk*, and *d* means *Severe Risk*. If a user is a control user, they automatically receive a *None* label, since we did not use crowdsourcing to label control users.

Column C (label) is label transforming the raw label into a ground truth value for binary classification.  Possible values are 1, 0, or -1. If the raw label of the user is *c* or *d*, the label is 1.  If the user is a control user, the label is -1 (by definition).  If the raw label is *a* or *b*, the label is 0. 

This transformation from raw labels into labels for binary classification is not the only way such a transformation could have been done. We have adopted the view that for purposes of binary classification, someone who posted to SuicideWatch should be considered truly a positive for suicidality screening if their risk level is moderate or severe (raw label c or d); but of course, one could cast a wider net by mapping low risk (raw label b)  to 1, or alternatively one could narrow the binary classification's focus by mapping moderate risk (raw label c) to 0.

Similarly, there are different ways one might choose to handle users labeled 0. The simplest assumption would be that users with label 0 should be excluded for purposes of binary classification, because they are neither positive instances, nor are they controls. 


### Expert

`expert.csv`: Annotation by experts. 245 users who posted on SuicideWatch and 245 control users are included. 

The labeling convention here is the same as for crowd.csv, except that raw_label was assigned by having four suicide assessment experts make judgments independently, and then assigning a consensus label using the Dawid-Skene  (1979) model for discovering true item states/effects from multiple noisy measurements; see  Rebecca J. Passonneau, Bob Carpenter, “The Benefits of a Model of Annotation”, TACL, 2014.
