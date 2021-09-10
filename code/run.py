import time, sys
import inspect
import fire

import advice_classifier
import augmented_advice_classifier

class Client:
    def advice_classifier(self, task="evaluate_and_error_analysis"):
        sig = inspect.signature(self.advice_classifier)
        for param in sig.parameters.values():
            print(f'{param.name:15s} = {eval(param.name)}')

        obj = advice_classifier.AdviceClassifier()
        try:
            func = getattr(obj, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()

    def augmented_advice_classifier(self, task='evaluate_and_error_analysis', feature_section=1, feature_citation=1, feature_past_tense=1):
        sig = inspect.signature(self.augmented_advice_classifier)
        for param in sig.parameters.values():
            print(f'{param.name:15s} = {eval(param.name)}')

        clf = augmented_advice_classifier.AugmentedAdviceClassifier(feature_section, feature_citation, feature_past_tense)
        try:
            func = getattr(clf, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()


if __name__ == "__main__":
    tic = time.time()
    fire.Fire(Client)
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')