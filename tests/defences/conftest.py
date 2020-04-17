import pytest
import logging

#TODO this needs to be refactored with get_image_classifier_list_for_attack in test/attacks/conftest.py
@pytest.fixture
def get_image_classifier_list_for_attack(get_image_classifier_list):
    def get_image_classifier_list_for_attack(attack, **kwargs):

        classifier_list, _ = get_image_classifier_list()
        if classifier_list is None:
            return None

        return [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

    return get_image_classifier_list_for_attack