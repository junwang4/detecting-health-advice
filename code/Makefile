quick: regular augmented
#quick: regular
#quick: augmented

regular:
	python run.py advice_classifier --task=train_KFold_model
	python run.py advice_classifier --task=evaluate_and_error_analysis
	python run.py advice_classifier --task=apply_trained_model_to_discussion_sentences
	python run.py advice_classifier --task=apply_trained_model_to_unstructured_abstract_sentences
	python run.py advice_classifier --task=postprocessing_filter_for_discussions
	python run.py advice_classifier --task=postprocessing_filter_for_unstructured_abstracts


feature_section = 1
feature_citation = 1
feature_past_tense = 1

augmented:
	python run.py augmented_advice_classifier --task=train_KFold_model --feature_section=$(feature_section) --feature_citation=$(feature_citation) --feature_past_tense=$(feature_past_tense)
	python run.py augmented_advice_classifier --task=evaluate_and_error_analysis --feature_section=$(feature_section) --feature_citation=$(feature_citation) --feature_past_tense=$(feature_past_tense)

