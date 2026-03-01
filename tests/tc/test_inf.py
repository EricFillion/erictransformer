"""
Tests for Text Classification Functionality
"""

from erictransformer import EricTextClassification
from tests.tc import (
    PROVIDED_LABELS,
    TC_CALL_TEXT,
    eric_tc,
    eric_tc_default_labels,
    eric_tc_provided_labels,
)


def test_inf_simple():
    result = eric_tc(TC_CALL_TEXT)
    assert type(result.labels[0]) == str
    assert type(result.scores[0]) == float



def test_label_types():

    result = eric_tc_default_labels("That song was so great!")
    assert result.labels[0]== "music"
    assert len(result.labels) == 19

    # Case 2: provide labels
    result = eric_tc_provided_labels("random text, either label either is fine")

    assert result.labels[0] in PROVIDED_LABELS

    # Case 3: num labels
    eric_tc_labels = EricTextClassification(
        model_name="prajjwal1/bert-tiny", labels=["LABEL_0"]
    )
    result = eric_tc_labels("random text")

    print(result)

    assert result.labels[0] == "LABEL_0"
